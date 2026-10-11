"""Independent robot/environment diagnostic, never a GPT policy or dressing claim.

Uses privileged mesh geometry to choose targets and measure the hem, but uses
unmodified GenesisController.plan/set_control/step for all manipulation.
"""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw
from matplotlib.path import Path as PolygonPath
from scipy.spatial.transform import Rotation

from dressing_scene import prepare_assets, build_scene
from genesis_route import GenesisController
from sloth_lift import garment_openings


def classify_grasps(events, vertices, faces):
    """Label rest-surface orientation for offline diagnostics only."""
    import trimesh
    normals=trimesh.Trimesh(vertices,faces,process=False).vertex_normals
    classified=[]
    for event in events:
        if event['event']!='attached' or event.get('object')!='garment':continue
        indices=np.concatenate(event['indices']).astype(int)
        front=indices[normals[indices,2]>.5].tolist()
        back=indices[normals[indices,2]<-.5].tolist()
        classified.append({'side':event['side'],'time_s':event['time'],
            'front_surface_indices':front,'back_surface_indices':back,
            'mixed_front_back_grasp':bool(front and back)})
    return classified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--grasp-x',type=float,default=.098,help='First hem pinch world X, metres')
    args=parser.parse_args()
    run=args.output.resolve();run.mkdir(parents=True,exist_ok=False)
    (run/'script.py').write_text(Path(__file__).read_text())
    config,paths,prepared,scene_report=prepare_assets(Path(__file__).resolve().parent.parent/'bunnylab',run)
    scene_args=SimpleNamespace(backend='cuda',substeps=10,particle_size=.004,width=640,height=480,headless=True,task='settle',rpc=True)
    report={'kind':'environment opening diagnostic; no GPT call',
            'uses_privileged_geometry':True,'uses_unmodified_web_controller':True,
            'direct_object_position_changes':False,'extra_particle_attachments':False,
            'opening_thresholds':{'projected_area_m2':.002,'inscribed_radius_m':.012,'hold_s':1.},
            'outcome':'incomplete','phases':[]}
    encoder=None;controller=None;trajectory=None
    try:
        scene,robots,objects,cameras=build_scene(config,paths,prepared,scene_report,scene_args)
        controller=GenesisController(scene,robots,objects,cameras,config,scene_report,run)
        physics=np.load(prepared/'physics_meshes.npz')
        rest=physics['garment_points'];faces=physics['garment_faces'];hem=garment_openings(rest,faces)['hem']
        np.savez(run/'reference.npz',vertices=rest,faces=faces,hem=hem)
        encoder=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-n','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1280x480','-r','10','-i','-','-an','-c:v','libx264','-preset','fast','-crf','23','-threads','2',
            '-pix_fmt','yuv420p','-movflags','+faststart',str(run/'opening.mp4')],stdin=subprocess.PIPE)
        trajectory=(run/'measurements.jsonl').open('w')
        phase='settle';samples=[];openings={'left':1.,'right':1.}
        orientations={side:Rotation.from_quat(np.array(controller.observe()['robot'][side+'_quaternion_wxyz'])[[1,2,3,0]]) for side in ('left','right')}
        def points():return objects['garment'].get_particles_pos().detach().cpu().numpy()
        def metric():
            p=points()[hem];center=p.mean(0);_,singular,basis=np.linalg.svd(p-center,full_matrices=False)
            xy=(p-center)@basis[:2].T
            ends=np.roll(xy,-1,axis=0);segments=ends-xy
            area=abs(np.sum(xy[:,0]*ends[:,1]-xy[:,1]*ends[:,0]))*.5
            def cross(a,b):return a[...,0]*b[...,1]-a[...,1]*b[...,0]
            simple=True
            for i in range(len(xy)):
                for j in range(i+2,len(xy)):
                    if i==0 and j==len(xy)-1:continue
                    if (cross(segments[i],xy[j]-xy[i])*cross(segments[i],ends[j]-xy[i])<0
                        and cross(segments[j],xy[i]-xy[j])*cross(segments[j],ends[i]-xy[j])<0):simple=False
            gx,gy=np.meshgrid(np.linspace(xy[:,0].min(),xy[:,0].max(),25),
                             np.linspace(xy[:,1].min(),xy[:,1].max(),15))
            grid=np.column_stack([gx.ravel(),gy.ravel()])
            grid=grid[PolygonPath(xy).contains_points(grid)]
            radius=0.
            if simple and len(grid):
                delta=grid[:,None,:]-xy
                t=np.clip(np.sum(delta*segments,axis=2)/np.maximum(np.sum(segments**2,axis=1),1e-16),0,1)
                radius=float(np.linalg.norm(delta-t[:,:,None]*segments,axis=2).min(axis=1).max())
            return {'time_s':controller.time,'phase':phase,'projected_area_m2':float(area),
                    'inscribed_radius_m':radius,'projection_simple':bool(simple),
                    'planarity_rms_m':float(singular[-1]/np.sqrt(len(p))),
                    'grasped':list(controller.attached),'robot':controller.observe()['robot'],'hem_world_m':p.tolist()}
        def sample(snapshot=False):
            m=metric();samples.append(m);trajectory.write(json.dumps(m)+'\n');trajectory.flush()
            rgb=[]
            for name in ('overhead','table_side'):
                im,*_=cameras[name].render();rgb.append(np.asarray(im))
            frame=Image.fromarray(np.concatenate(rgb,axis=1));draw=ImageDraw.Draw(frame)
            draw.rectangle((0,0,1280,40),fill='black');draw.text((10,8),f'{phase} | t={controller.time:.1f}s | hem area={m["projected_area_m2"]*1e4:.1f} cm2 | radius={m["inscribed_radius_m"]*1000:.1f} mm',fill='white')
            encoder.stdin.write(np.asarray(frame).tobytes())
            if snapshot:frame.save(run/(phase+'.png'));np.savez(run/(phase+'.npz'),garment=points())
        def advance(seconds):
            for _ in range(round(seconds/controller.dt/10)):
                controller.step(10);sample()
        def stage(name):
            nonlocal phase
            phase=name;report['phases'].append({'name':name,'time_s':controller.time});print(name,flush=True);sample(True)
        def move(targets,limit=90):
            for _ in range(limit):
                state=controller.observe()['robot'];action={};largest=0.
                for side in ('left','right'):
                    current=np.array(state[side+'_position_m']);delta=np.asarray(targets.get(side,current))-current
                    largest=max(largest,float(np.linalg.norm(delta)));delta*=min(1.,.015/max(np.linalg.norm(delta),1e-12))
                    current_rotation=Rotation.from_quat(np.array(state[side+'_quaternion_wxyz'])[[1,2,3,0]])
                    correction=(orientations[side]*current_rotation.inv()).as_rotvec()
                    correction*=min(1.,.08/max(np.linalg.norm(correction),1e-12))
                    action.update({side+'_delta_m':delta.tolist(),side+'_rotation_rad':correction.tolist(),side+'_gripper':openings[side]})
                if largest<.0015:return
                for scale in (1.,.5,.25,.125):
                    trial=dict(action)
                    for side in ('left','right'):
                        for suffix in ('_delta_m','_rotation_rad'):trial[side+suffix]=(np.array(action[side+suffix])*scale).tolist()
                    try:plan=controller.plan(trial);break
                    except ValueError:
                        if scale==.125:raise
                controller.set_control(plan['controls']);advance(.5)
            raise RuntimeError('Target tracking did not finish within 90 bounded actions')
        def gripper(side,value):
            openings[side]=value
            action={side+suffix:val for side in ('left','right') for suffix,val in
                    (('_delta_m',[0.,0.,0.]),('_rotation_rad',[0.,0.,0.]),('_gripper',openings[side]))}
            controller.set_control(controller.plan(action)['controls']);advance(.5)
        advance(1.)
        stage('approach_front_hem')
        front=int(hem[np.argmin(abs(rest[hem,0]-args.grasp_x))])
        target=points()[front].copy();target[2]=config['scene']['table_height']+.10
        right_pregrasp=np.array([rest[hem,0].max()-.01,rest[hem,1].mean(),target[2]])
        move({'left':target,'right':right_pregrasp})
        gripper('left',0.)
        stage('pinch_front_hem')
        for _ in range(30):
            target=points()[front].copy();target[2]=max(config['scene']['table_height']+.012,
                controller.observe()['robot']['left_position_m'][2]-.004)
            move({'left':target});advance(.1)
            if 'left' in controller.attached:break
        if 'left' not in controller.attached:raise RuntimeError('Front hem pinch failed with the existing contact gate')
        stage('peel_front_hem')
        target=np.array(controller.observe()['robot']['left_position_m']);target[2]+=.07
        move({'left':target});advance(1.)
        stage('approach_back_hem')
        # Select a low, unbound hem point accessible from the right, after peeling.
        eligible=[int(i) for i in hem if ('garment',int(i)) not in controller.used and points()[i,0]>.19]
        if not eligible:raise RuntimeError('No free back hem target')
        back=min(eligible,key=lambda i:float(points()[i,2]))
        target=points()[back].copy();target[2]=max(target[2]+.08,config['scene']['table_height']+.10)
        move({'right':target});gripper('right',0.)
        stage('pinch_back_hem')
        for _ in range(30):
            target=points()[back].copy();target[2]=max(config['scene']['table_height']+.012,
                controller.observe()['robot']['right_position_m'][2]-.004)
            move({'right':target});advance(.1)
            if 'right' in controller.attached:break
        if 'right' not in controller.attached:raise RuntimeError('Back hem pinch failed with the existing contact gate')
        stage('separate_and_hold')
        target=np.array(controller.observe()['robot']['left_position_m']);target[2]+=.04;target[1]+=.025
        move({'left':target});advance(1.5)
        stage('final')
        last=[m for m in samples if m['time_s']>=controller.time-1.]
        report['opening_held']=all(m['projection_simple'] and m['projected_area_m2']>.002 and m['inscribed_radius_m']>.012 for m in last)
        report['outcome']='opening_observed' if report['opening_held'] else 'opening_not_demonstrated'
    except Exception as exc:
        report['outcome']='attempt_failed';report['error']=str(exc)
        print('Diagnostic stopped:',exc,flush=True)
    finally:
        if controller:
            if 'sample' in locals():
                try:sample(True)
                except Exception as capture_error:report['capture_error']=str(capture_error)
            report['simulation_seconds']=controller.time
            report['grasp_events']=[json.loads(line) for line in (run/'grasp_events.jsonl').read_text().splitlines()]
            report['grasp_surface_diagnostics']=classify_grasps(report['grasp_events'],rest,faces)
            if 'samples' in locals() and samples:
                report['initial']=samples[0];report['final']=samples[-1]
                report['maximum_inscribed_radius_m']=max(m['inscribed_radius_m'] for m in samples)
            controller.contact_log.close()
        if trajectory:trajectory.close()
        if encoder:encoder.stdin.close();report['video_encoder_exit_code']=encoder.wait()
        report['limitations']=['Privileged scripted controller: does not evaluate GPT planning.',
            'A projected hem opening alone does not certify a collision-free 3D passage or completed dressing.',
            'Existing no-slip contact attachments are reused; real frictional grasp force is not validated.',
            'Script uses 0.5-second feedback; web policy currently observes after five 1-second chunks.']
        (run/'scene-report.json').write_text(json.dumps(scene_report,indent=2))
        (run/'report.json').write_text(json.dumps(report,indent=2))
        import genesis as gs
        gs.destroy()
        print(json.dumps({'outcome':report['outcome'],'output':str(run)},indent=2),flush=True)

if __name__=='__main__':main()
