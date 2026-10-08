"""Small, version-checked adapter for pre-existing PBD triangle/tetra meshes.

Genesis 1.4.3's Mesh importer always remeshes and retetrahedralizes PBD assets.
The bunnylab tetra mesh has a non-manifold exterior edge, so that path cannot
preserve it. The import context reuses Genesis's PBD constraints and state APIs,
replacing only pre-build sampling with the supplied vertices/connectivity.
No installed Genesis files or global disk caches are modified.
"""
from contextlib import contextmanager
from importlib.metadata import version

import numpy as np
import trimesh


def simulation_connectivity(entity, kind):
    """Connectivity for the adapter's import-fidelity check (private 1.4.3 field)."""
    return entity._elems if kind == "sloth" else entity.mesh.faces


def install_rigid_pbd_friction(scene):
    """Enable Coulomb contact friction on this scene, before kernel compilation.

    Genesis 1.4.3's legacy contact performs only normal SDF projection. Use
    position-level friction (Macklin et al. 2014, section 6.1), consistent with
    Genesis's PBD particle contacts. The normal projection is the friction
    budget; there is no attraction, world-position pinning, or velocity reset.
    Both coefficients use the smaller of the particle and rigid coefficients.
    Attached particles retain the existing contact/attachment behavior.
    """
    import quadrants as qd
    import genesis as gs
    import genesis.utils.sdf as sdf
    from genesis.engine.couplers.legacy_coupler import LegacyCoupler
    from genesis.utils import array_class

    if version("genesis-world") != "1.4.3":
        raise RuntimeError("Rigid/PBD friction requires the reviewed Genesis 1.4.3 contact interface")
    original = scene.sim.coupler
    if type(original) is not LegacyCoupler or scene.is_built:
        raise RuntimeError("Install rigid/PBD friction once, on an unbuilt legacy-coupler scene")

    @qd.data_oriented
    class FrictionCoupler(LegacyCoupler):
        @qd.func
        def _func_pbd_collide_with_rigid_geom(
            self, i, pos_world, vel, mass, normal_prev, geom_idx, batch_idx,
            geoms_state: array_class.GeomsState,
            geoms_info: array_class.GeomsInfo,
            links_state: array_class.LinksState,
            sdf_info: array_class.SDFInfo,
            rigid_info: array_class.RigidInfo,
            collider_static_config: qd.template(),
        ):
            distance = sdf.sdf_func_world(geom_idx, batch_idx, pos_world, geoms_state, geoms_info, sdf_info)
            normal = sdf.sdf_func_normal_world(
                geom_idx, batch_idx, pos_world, geoms_state, geoms_info,
                rigid_info, sdf_info, collider_static_config)
            new_pos, new_vel = pos_world, vel
            depth = self.pbd_solver.particle_size / 2 - distance
            if depth > 0:
                new_pos = pos_world + normal * depth
                previous = self.pbd_solver.particles_reordered[i, batch_idx].ipos
                if self.pbd_solver.particles_reordered[i, batch_idx].free:
                    rigid_velocity = self.rigid_solver._func_vel_at_point(
                        pos_world=pos_world, link_idx=geoms_info.link_idx[geom_idx],
                        i_b=batch_idx, links_state=links_state)
                    relative_delta = new_pos - previous - rigid_velocity * self.pbd_solver._substep_dt
                    tangent = relative_delta - normal * relative_delta.dot(normal)
                    tangent_length = tangent.norm()
                    mu_s = qd.min(self.pbd_solver.particles_info_reordered[i, batch_idx].mu_s,
                                  geoms_info.coup_friction[geom_idx])
                    mu_k = qd.min(self.pbd_solver.particles_info_reordered[i, batch_idx].mu_k,
                                  geoms_info.coup_friction[geom_idx])
                    if tangent_length > 1e-12:
                        fraction = 1.0
                        if tangent_length > mu_s * depth:
                            fraction = qd.min(1.0, mu_k * depth / tangent_length)
                        new_pos -= fraction * tangent
                new_vel = (new_pos - previous) / self.pbd_solver._substep_dt
                # Preserve the original equal-and-opposite rigid coupling force,
                # including the tangential impulse introduced by friction.
                force = -mass * (new_vel - vel) / self.rigid_solver._substep_dt
                self.rigid_solver._func_apply_coupling_force(
                    geoms_info.link_idx[geom_idx], batch_idx, pos_world, force, links_state)
            return new_pos, new_vel, normal

    scene.sim._coupler = FrictionCoupler(scene.sim, original.options)
    return {"adapter": "scene-local Genesis 1.4.3 legacy Coulomb friction",
            "method": "position-level static/kinetic friction limited by normal SDF projection",
            "coefficient_combination": "minimum(particle coefficient, rigid coup_friction)",
            "applies_to": "free PBD particles contacting coupled rigid geometry",
            "normal_projection_unchanged": True, "reciprocal_rigid_force": True,
            "world_position_pinning": False, "installed_genesis_modified": False}


@contextmanager
def existing_pbd_mesh(kind, points, elements, mass):
    import genesis as gs
    import genesis.engine.solvers.pbd_solver as solver_module

    if version("genesis-world") != "1.4.3":
        raise RuntimeError("The existing-mesh adapter was verified with Genesis 1.4.3; review it before using another version.")
    class_name = "PBD3DEntity" if kind == "sloth" else "PBD2DEntity"
    original_class = getattr(solver_module, class_name)

    def sample(self):
        # Files have already been transformed into world coordinates.
        if tuple(self.morph.pos) != (0., 0., 0.) or tuple(self.morph.quat) != (1., 0., 0., 0.):
            raise ValueError("Existing PBD mesh must use an identity morph transform")
        self._sampled_pos = np.zeros(3, dtype=gs.np_float)
        self._vverts = np.asarray(self._vmesh.verts, dtype=gs.np_float)
        self._vfaces = np.asarray(self._vmesh.faces, dtype=gs.np_int)
        self._mesh = self._vmesh.copy()
        self._particles = np.asarray(points, dtype=gs.np_float)
        self._init_particles_offset = gs.tensor(self._particles)
        self._mass = float(mass)
        self._n_particles = len(self._particles)
        self._particle_mass = self._mass / self._n_particles
        elems = np.asarray(elements, dtype=gs.np_int)
        if kind == "sloth":
            self._elems = elems.copy()
            volumes = np.linalg.det(self._particles[elems[:, 1:]] - self._particles[elems[:, :1]]) / 6
            if np.any(np.abs(volumes) < 1e-14):
                raise ValueError("Source sloth mesh contains degenerate tetrahedra")
            # Normalize orientation; vertex sets and tetra connectivity stay identical.
            negative = volumes < 0
            self._elems[negative] = self._elems[negative][:, [0, 2, 1, 3]]
            self._elems_vol_rest = np.abs(volumes)
            self._edges = np.unique(np.sort(np.concatenate([
                self._elems[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), axis=1), axis=0)
        else:
            self._mesh = gs.Mesh.from_trimesh(trimesh.Trimesh(
                vertices=self._particles, faces=elems, process=False), surface=self.surface)
            self._edges = np.unique(np.sort(np.concatenate([
                elems[:, [0, 1]], elems[:, [1, 2]], elems[:, [2, 0]]]), axis=1), axis=0)
            adjacency, shared = trimesh.graph.face_adjacency(mesh=self._mesh.trimesh, return_edges=True)
            opposite1 = elems[adjacency[:, 0]].sum(axis=1) - shared.sum(axis=1)
            opposite2 = elems[adjacency[:, 1]].sum(axis=1) - shared.sum(axis=1)
            self._inner_edges = np.column_stack([shared, opposite1, opposite2]).astype(gs.np_int)
            self._inner_edges_len_rest = np.linalg.norm(
                self._particles[opposite1] - self._particles[opposite2], axis=1)
        self._edges = self._edges.astype(gs.np_int, copy=False)
        self._edges_len_rest = np.linalg.norm(
            self._particles[self._edges[:, 0]] - self._particles[self._edges[:, 1]], axis=1)
    # Keep the original entity class: solver counts use isinstance(), so
    # temporarily replacing that class would miss previously imported meshes.
    original_sample = original_class.sample
    original_class.sample = sample
    try:
        yield
    finally:
        original_class.sample = original_sample
