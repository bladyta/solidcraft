//! Surface Fillet (`surface.fillet`, #769): a rolling ball between two faces of separate bodies.
//!
//! The ball's radius is given. The pick on each face chooses the side the ball rolls on. Where
//! the ball touches both faces, the blend is the canal of that ball: a cylinder when both faces
//! are planes, or when a cylinder's axis is parallel to a plane, and a torus when the axis is
//! square to the plane. The faces are trimmed back to the contact and sewn to the blend, or the
//! blend is returned alone.
//!
//! Anything else (two cylinders, a tilted axis, a free-form face) is refused. A ball that cannot
//! touch both faces is refused too, and nothing here panics.

use mt::{BoundedCurve, ParametricCurve, ParametricSurface};
use solidcraft_geom::{Mesh, Plane, Vec3};
use truck_modeling as mt;

use crate::body::{from_p3, p3, v3};
use crate::surface_kind::{FaceSurface, face_surface};
use crate::surfaces::surface_body;
use crate::{Body, KernelError, Result, SplitTool, guard, stitch, trim};

/// Which of the two bodies the fillet trims away as it sews the blend on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Trim {
    /// Both bodies are trimmed and sewn to the blend. One body comes back.
    #[default]
    Both,
    /// Only the first body is trimmed and sewn to the blend. The second is untouched.
    First,
    /// The blend alone. Both bodies stay as they were.
    None,
}

/// A plane the trim keeps the positive side of. `normal` points at the side that stays.
struct Cut {
    at: Vec3,
    normal: Vec3,
}

/// The blend, and a cut for each body (none when the contact is not a straight line).
struct Fillet {
    blend: Body,
    cut_a: Option<Cut>,
    cut_b: Option<Cut>,
}

struct Located {
    face: usize,
    surf: FaceSurface,
    mesh: Mesh,
}

const R_MIN: f64 = 1e-6;
const R_MAX: f64 = 1e6;

/// The rolling-ball fillet of radius `radius` between the face of `a` nearest `pa` and the face
/// of `b` nearest `pb`. The points pick the side. `trim` decides whether those bodies are cut
/// back to the contact and sewn on.
pub fn surface_fillet(a: &Body, pa: Vec3, b: &Body, pb: Vec3, radius: f64, trim_mode: Trim) -> Result<Body> {
    if !(radius.is_finite() && radius > R_MIN && radius < R_MAX) {
        return Err(KernelError::Invalid(format!("radius must be between {R_MIN} and {R_MAX}")));
    }
    if !pa.is_finite() || !pb.is_finite() {
        return Err(KernelError::Invalid("the pick points must be finite".into()));
    }
    if trim_mode != Trim::None && (!a.is_surface() || !b.is_surface()) {
        return Err(KernelError::Invalid("not supported yet: trimming a solid (use trim none)".into()));
    }
    let la = locate(a, pa)?;
    let lb = locate(b, pb)?;
    let built = guard("surface fillet", || match (&la.surf, &lb.surf) {
        (FaceSurface::Plane { .. }, FaceSurface::Plane { .. }) => plane_plane(a, &la, pa, b, &lb, pb, radius),
        (FaceSurface::Plane { .. }, FaceSurface::Cylinder { .. }) => plane_cylinder(a, &la, pa, b, &lb, pb, radius, true),
        (FaceSurface::Cylinder { .. }, FaceSurface::Plane { .. }) => plane_cylinder(b, &lb, pb, a, &la, pa, radius, false),
        (FaceSurface::Plane { .. }, _) | (_, FaceSurface::Plane { .. }) | (FaceSurface::Cylinder { .. }, FaceSurface::Cylinder { .. }) => {
            Err(later("a rolling-ball fillet between these surfaces (a plane and a plane, or a plane and a cylinder)"))
        }
        _ => Err(later("a rolling-ball fillet on a face that is not a plane or a cylinder")),
    })?;
    apply_trim(a, b, trim_mode, built)
}

fn later(what: &str) -> KernelError {
    KernelError::Failed(format!("not supported yet: {what}"))
}

fn invalid(what: &str) -> KernelError {
    KernelError::Invalid(what.into())
}

fn locate(body: &Body, p: Vec3) -> Result<Located> {
    let mesh = body.tessellate((body.size() * 2e-4).max(1e-3))?;
    let tol = on_tol(body);
    let face = nearest_on(&mesh, p, tol).ok_or_else(|| invalid("the point is not on a face"))?;
    let surf = face_surface(body, face).ok_or_else(|| later("a rolling-ball fillet on a face that is not a plane, a cylinder, a cone, a sphere or a torus"))?;
    if analytic_gap(&surf, p) > tol {
        return Err(invalid("the point is not on a face"));
    }
    Ok(Located { face, surf, mesh })
}

fn on_tol(body: &Body) -> f64 {
    (body.size() * 1e-3).max(1e-2)
}

fn nearest_on(mesh: &Mesh, p: Vec3, tol: f64) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        let Some([a, b, c]) = mesh.tri(t) else { continue };
        let d = solidcraft_geom::compare::closest_on_triangle(p, a, b, c).dist(p);
        if best.is_none_or(|(x, _)| d < x) {
            best = Some((d, *f as usize));
        }
    }
    best.filter(|(d, _)| *d <= tol).map(|(_, f)| f)
}

fn on_face(loc: &Located, p: Vec3, tol: f64) -> bool {
    let mut best = f64::MAX;
    for (t, f) in loc.mesh.triangles.iter().zip(&loc.mesh.tri_face) {
        if *f as usize != loc.face {
            continue;
        }
        let Some([a, b, c]) = loc.mesh.tri(t) else { continue };
        best = best.min(solidcraft_geom::compare::closest_on_triangle(p, a, b, c).dist(p));
    }
    best <= tol
}

fn analytic_gap(s: &FaceSurface, p: Vec3) -> f64 {
    match *s {
        FaceSurface::Plane { normal, point } => (p - point).dot(normal).abs(),
        FaceSurface::Cylinder { origin, axis, radius } => {
            let d = p - origin;
            (d - axis * d.dot(axis)).len() - radius
        }
        FaceSurface::Sphere { center, radius } => p.dist(center) - radius,
        FaceSurface::Cone { apex, axis, .. } => (p - apex).cross(axis).len(),
        FaceSurface::Torus { center, axis, major, minor } => {
            let d = p - center;
            let axial = d.dot(axis);
            let radial = (d - axis * axial).len();
            (radial - major).hypot(axial) - minor
        }
    }
    .abs()
}

fn apply_trim(a: &Body, b: &Body, mode: Trim, fillet: Fillet) -> Result<Body> {
    if mode == Trim::None {
        return Ok(fillet.blend);
    }
    let (Some(ca), Some(cb)) = (fillet.cut_a, fillet.cut_b) else {
        return Err(invalid("not supported yet: trimming this fillet (use trim none)"));
    };
    let ta = cut_body(a, &ca)?;
    if mode == Trim::First {
        return stitch(&[&ta, &fillet.blend], 1e-4);
    }
    let tb = cut_body(b, &cb)?;
    stitch(&[&ta, &tb, &fillet.blend], 1e-4)
}

fn cut_body(body: &Body, cut: &Cut) -> Result<Body> {
    let n = cut.normal.normalized().ok_or_else(|| invalid("a fillet with no side"))?;
    let plane = Plane::from_normal(cut.at, n).ok_or_else(|| invalid("a fillet with no side"))?;
    trim(body, &SplitTool::Plane(plane), cut.at + n)
}

fn unit(v: Vec3, what: &str) -> Result<Vec3> {
    v.normalized().ok_or_else(|| invalid(what))
}

fn side_sign(d: f64, tol: f64) -> Result<f64> {
    if d.abs() <= tol {
        Err(invalid("the point doesn't pick a side"))
    } else if d > 0.0 {
        Ok(1.0)
    } else {
        Ok(-1.0)
    }
}

/// Signed turn from `from` to `to` about `axis` (radians, in (−π, π]).
fn signed_turn(axis: Vec3, from: Vec3, to: Vec3) -> f64 {
    let from = from - axis * from.dot(axis);
    let to = to - axis * to.dot(axis);
    axis.dot(from.cross(to)).atan2(from.dot(to))
}

fn boundary_points(body: &Body, face: usize) -> Result<Vec<Vec3>> {
    let f = body.solid.face_iter().nth(face).ok_or_else(|| invalid("no face"))?;
    let mut pts = Vec::new();
    for w in f.boundaries().iter() {
        for e in w.edge_iter() {
            let c = e.oriented_curve();
            let (t0, t1) = c.range_tuple();
            for i in 0..=16 {
                let t = t0 + (t1 - t0) * f64::from(i) / 16.0;
                pts.push(from_p3(c.subs(t)));
            }
        }
    }
    if pts.is_empty() {
        return Err(invalid("the face has no boundary"));
    }
    Ok(pts)
}

fn project_span(pts: &[Vec3], origin: Vec3, axis: Vec3) -> (f64, f64) {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for p in pts {
        let s = (*p - origin).dot(axis);
        lo = lo.min(s);
        hi = hi.max(s);
    }
    (lo, hi)
}

fn overlap(a: &[Vec3], b: &[Vec3], origin: Vec3, axis: Vec3, tol: f64) -> Result<(f64, f64)> {
    let (a0, a1) = project_span(a, origin, axis);
    let (b0, b1) = project_span(b, origin, axis);
    let lo = a0.max(b0);
    let hi = a1.min(b1);
    if !(hi - lo > tol) {
        return Err(invalid("the faces don't meet or cross"));
    }
    Ok((lo, hi))
}

/// Samples of the segment must lie on the face. Endpoints sit on the boundary, so the tolerance
/// is the on-face one.
fn segment_on(loc: &Located, p0: Vec3, p1: Vec3, tol: f64) -> bool {
    (0..=4).all(|i| on_face(loc, p0 + (p1 - p0) * (f64::from(i) / 4.0), tol))
}

fn plane_of(s: &FaceSurface) -> Result<(Vec3, Vec3)> {
    match *s {
        FaceSurface::Plane { normal, point } => Ok((unit(normal, "a plane with no normal")?, point)),
        _ => Err(invalid("not a plane")),
    }
}

fn cylinder_of(s: &FaceSurface) -> Result<(Vec3, Vec3, f64)> {
    match *s {
        FaceSurface::Cylinder { origin, axis, radius } => Ok((origin, unit(axis, "a cylinder with no axis")?, radius)),
        _ => Err(invalid("not a cylinder")),
    }
}

fn plane_plane(a: &Body, la: &Located, pa: Vec3, b: &Body, lb: &Located, pb: Vec3, r: f64) -> Result<Fillet> {
    let (n1, p1) = plane_of(&la.surf)?;
    let (n2, p2) = plane_of(&lb.surf)?;
    let tol = on_tol(a).max(on_tol(b));
    let g = n1.dot(n2).clamp(-1.0, 1.0);
    let det = 1.0 - g * g;
    if det < 1e-8 {
        let dist = (p2 - p1).dot(n1).abs();
        if dist <= tol {
            return Err(invalid("the faces lie on the same plane"));
        }
        if (dist - 2.0 * r).abs() <= tol.max(r * 1e-6) {
            return Err(invalid("the ball touches both but doesn't trace one curve"));
        }
        return Err(invalid("the ball can't touch both"));
    }
    // The other pick chooses the side: its signed distance from this plane.
    let s1 = side_sign((pb - p1).dot(n1), tol)?;
    let s2 = side_sign((pa - p2).dot(n2), tol)?;
    let (c0, along) = spine_of(n1, p1, n2, p2, g, det, s1, s2, r)?;
    let radial0 = n1 * -s1;
    let radial1 = n2 * -s2;
    let mut axis = along;
    let mut sweep = signed_turn(axis, radial0, radial1);
    if sweep < 0.0 {
        axis = -axis;
        sweep = -sweep;
    }
    if sweep < 1e-4 {
        return Err(invalid("the fillet is too flat to build"));
    }
    let pts_a = boundary_points(a, la.face)?;
    let pts_b = boundary_points(b, lb.face)?;
    let (t0, t1) = overlap(&pts_a, &pts_b, c0, axis, tol)?;
    let c_lo = c0 + axis * t0;
    let c_hi = c0 + axis * t1;
    let a0 = c_lo + radial0 * r;
    let a1 = c_hi + radial0 * r;
    let b0 = c_lo + radial1 * r;
    let b1 = c_hi + radial1 * r;
    if !segment_on(la, a0, a1, tol) || !segment_on(lb, b0, b1, tol) {
        return Err(invalid("the faces don't meet or cross"));
    }
    let blend = revolved_line(c_lo, axis, a0, a1, sweep)?;
    // The cut runs through the contact. Its normal points from the planes' intersection, through
    // the contact, and on: that is the side the sheet keeps.
    let (i0, _) = spine_of(n1, p1, n2, p2, g, det, 0.0, 0.0, 0.0)?;
    Ok(Fillet {
        blend,
        cut_a: Some(Cut { at: a0, normal: keep_dir(i0, a0, axis)? }),
        cut_b: Some(Cut { at: b0, normal: keep_dir(i0, b0, axis)? }),
    })
}

/// A point on the spine `c0` and a unit direction along it. `s1`/`s2` are ±1 (0 builds the
/// intersection of the two planes, used as the trim's reference).
fn spine_of(n1: Vec3, p1: Vec3, n2: Vec3, p2: Vec3, g: f64, det: f64, s1: f64, s2: f64, r: f64) -> Result<(Vec3, Vec3)> {
    let rhs1 = s1 * r + p1.dot(n1);
    let rhs2 = s2 * r + p2.dot(n2);
    let alpha = (rhs1 - g * rhs2) / det;
    let beta = (rhs2 - g * rhs1) / det;
    let c0 = n1 * alpha + n2 * beta;
    let along = unit(n1.cross(n2), "the planes are parallel")?;
    if !c0.is_finite() {
        return Err(invalid("the ball can't touch both"));
    }
    Ok((c0, along))
}

fn keep_dir(from: Vec3, contact: Vec3, along: Vec3) -> Result<Vec3> {
    let d = contact - from;
    let d = d - along * d.dot(along);
    unit(d, "the fillet meets the face on the intersection")
}

fn plane_cylinder(
    plane_body: &Body,
    plane_loc: &Located,
    plane_pick: Vec3,
    cyl_body: &Body,
    cyl_loc: &Located,
    cyl_pick: Vec3,
    r: f64,
    plane_is_a: bool,
) -> Result<Fillet> {
    let (n, plane_pt) = plane_of(&plane_loc.surf)?;
    let (origin, axis, radius) = cylinder_of(&cyl_loc.surf)?;
    let tol = on_tol(plane_body).max(on_tol(cyl_body));
    let align = axis.dot(n).abs();
    if align > 1.0 - 1e-6 {
        let fillet = torus_fillet(plane_body, plane_loc, plane_pick, cyl_body, cyl_loc, cyl_pick, n, plane_pt, origin, axis, radius, r, tol)?;
        return Ok(ordered(fillet, plane_is_a));
    }
    if align > 1e-6 {
        return Err(later("a cylinder whose axis is neither parallel nor square to the plane"));
    }
    let fillet = parallel_fillet(plane_body, plane_loc, plane_pick, cyl_body, cyl_loc, cyl_pick, n, plane_pt, origin, axis, radius, r, tol)?;
    Ok(ordered(fillet, plane_is_a))
}

fn ordered(f: Fillet, plane_is_a: bool) -> Fillet {
    if plane_is_a { f } else { Fillet { cut_a: f.cut_b, cut_b: f.cut_a, ..f } }
}

struct Cand {
    /// Signed offset of the spine from the axis, along `lateral`.
    offset: f64,
    lateral: Vec3,
    external: bool,
    s_p: f64,
    score: f64,
}

fn parallel_fillet(
    plane_body: &Body,
    plane_loc: &Located,
    plane_pick: Vec3,
    cyl_body: &Body,
    cyl_loc: &Located,
    cyl_pick: Vec3,
    n: Vec3,
    plane_pt: Vec3,
    origin: Vec3,
    axis: Vec3,
    radius: f64,
    r: f64,
    tol: f64,
) -> Result<Fillet> {
    let lateral = unit(axis.cross(n), "the cylinder is square to the plane")?;
    let delta = (origin - plane_pt).dot(n);
    let height = (cyl_pick - plane_pt).dot(n);
    let sides: Vec<f64> = if height.abs() <= tol { vec![1.0, -1.0] } else { vec![height.signum()] };
    let mut cands = Vec::new();
    for s_p in sides {
        let delta_c = delta - s_p * r;
        for (rho, external) in [(radius + r, true), ((radius - r).abs(), false)] {
            if rho < 1e-6 || delta_c.abs() > rho + 1e-8 {
                continue;
            }
            let reach = (rho * rho - delta_c * delta_c).max(0.0).sqrt();
            let signs: Vec<f64> = if reach < 1e-8 { vec![1.0] } else { vec![1.0, -1.0] };
            for sign in signs {
                let offset = sign * reach;
                let center = origin - n * delta_c + lateral * offset;
                let radial = center - origin;
                let Some(radial_u) = radial.normalized() else { continue };
                let plane_at = center - n * s_p * r;
                let cyl_at = if external { center - radial_u * r } else { center + radial_u * r };
                let score = line_dist(plane_pick, plane_at, axis) + line_dist(cyl_pick, cyl_at, axis);
                cands.push(Cand { offset, lateral, external, s_p, score });
            }
        }
    }
    if cands.is_empty() {
        return Err(invalid("the ball can't touch both"));
    }
    cands.sort_by(|a, b| a.score.total_cmp(&b.score));
    let mut last = invalid("the ball can't touch both");
    for c in &cands {
        match parallel_one(plane_body, plane_loc, cyl_body, cyl_loc, n, origin, axis, r, tol, c) {
            Ok(f) => return Ok(f),
            Err(e) => last = e,
        }
    }
    Err(last)
}

fn line_dist(p: Vec3, origin: Vec3, axis: Vec3) -> f64 {
    let d = p - origin;
    (d - axis * d.dot(axis)).len()
}

fn parallel_one(
    plane_body: &Body,
    plane_loc: &Located,
    cyl_body: &Body,
    cyl_loc: &Located,
    n: Vec3,
    origin: Vec3,
    axis: Vec3,
    r: f64,
    tol: f64,
    c: &Cand,
) -> Result<Fillet> {
    let delta = (origin - plane_of(&plane_loc.surf)?.1).dot(n);
    let delta_c = delta - c.s_p * r;
    let center = origin - n * delta_c + c.lateral * c.offset;
    let radial_u = unit(center - origin, "the ball centre is on the axis")?;
    let to_plane = n * -c.s_p;
    let to_cyl = if c.external { radial_u * -1.0 } else { radial_u };
    let mut spin = axis;
    let mut sweep = signed_turn(spin, to_plane, to_cyl);
    if sweep < 0.0 {
        spin = -spin;
        sweep = -sweep;
    }
    if sweep < 1e-4 {
        return Err(invalid("the fillet is too flat to build"));
    }
    let pts_p = boundary_points(plane_body, plane_loc.face)?;
    let pts_c = boundary_points(cyl_body, cyl_loc.face)?;
    let (t0, t1) = overlap(&pts_p, &pts_c, origin, spin, tol)?;
    let t_c = (center - origin).dot(spin);
    let at = |t: f64| center + spin * (t - t_c);
    let p0 = at(t0) + to_plane * r;
    let p1 = at(t1) + to_plane * r;
    let q0 = at(t0) + to_cyl * r;
    let q1 = at(t1) + to_cyl * r;
    if !segment_on(plane_loc, p0, p1, tol) || !segment_on(cyl_loc, q0, q1, tol) {
        return Err(invalid("the faces don't meet or cross"));
    }
    let blend = revolved_line(at(t0), spin, p0, p1, sweep)?;
    // Plane: keep the side past the contact, away from the axis. Cylinder: keep the side of the
    // generator the sheet continues on, away from the plane.
    let foot = origin - n * delta;
    let plane_cut = Cut { at: p0, normal: keep_dir(foot, p0, spin)? };
    let tangent = {
        let t = spin.cross(radial_u);
        if t.dot(n) * c.s_p < 0.0 { -t } else { t }
    };
    let cyl_cut = Cut { at: q0, normal: unit(tangent, "a cylinder fillet with no side")? };
    Ok(Fillet { blend, cut_a: Some(plane_cut), cut_b: Some(cyl_cut) })
}

fn torus_fillet(
    _plane_body: &Body,
    plane_loc: &Located,
    plane_pick: Vec3,
    cyl_body: &Body,
    cyl_loc: &Located,
    cyl_pick: Vec3,
    n: Vec3,
    plane_pt: Vec3,
    origin: Vec3,
    axis: Vec3,
    radius: f64,
    r: f64,
    tol: f64,
) -> Result<Fillet> {
    // The axis meets the plane at one point. The pick's height chooses the side; how far the
    // plane pick sits from the axis chooses an outside ball (major R+r) or an inside one.
    let den = axis.dot(n);
    if den.abs() < 1e-8 {
        return Err(invalid("the ball can't touch both"));
    }
    let foot = origin + axis * ((plane_pt - origin).dot(n) / den);
    let height = (cyl_pick - plane_pt).dot(n);
    let s_p = side_sign(height, tol)?;
    let x_pick = plane_pick - foot - n * (plane_pick - foot).dot(n);
    let outside = x_pick.len() > radius + tol;
    let rho = if outside { radius + r } else { (radius - r).abs() };
    if rho < 1e-6 {
        return Err(invalid("the ball can't touch both"));
    }
    let pts = boundary_points(cyl_body, cyl_loc.face)?;
    let frame_x = unit(axis.any_perp(), "a cylinder with no axis")?;
    let frame_y = axis.cross(frame_x);
    let ang = |p: Vec3| {
        let d = p - foot;
        d.dot(frame_y).atan2(d.dot(frame_x))
    };
    let angles: Vec<f64> = pts.iter().copied().map(ang).collect();
    let (start, span) = angle_span(&angles).ok_or_else(|| invalid("the faces don't meet or cross"))?;
    if span < 1e-4 {
        return Err(invalid("the faces don't meet or cross"));
    }
    let center_at = |th: f64| foot + n * s_p * r + (frame_x * th.cos() + frame_y * th.sin()) * rho;
    let radial_at = |th: f64| frame_x * th.cos() + frame_y * th.sin();
    let plane_contact = |th: f64| center_at(th) - n * s_p * r;
    let cyl_contact = |th: f64| {
        let c = center_at(th);
        if outside { c - radial_at(th) * r } else { c + radial_at(th) * r }
    };
    let (start, span) = covered_span(plane_loc, cyl_loc, start, span, &plane_contact, &cyl_contact, tol)?;
    let pick_ang = ang(plane_pick);
    let _ = pick_ang;
    let profile = tube_arc(center_at(start), plane_contact(start), cyl_contact(start), r)?;
    let blend = revolved_profile(profile, foot, axis, span)?;
    // The contact on the plane is an arc, so one plane cannot trim it.
    Ok(Fillet { blend, cut_a: None, cut_b: None })
}

/// The run of `start .. start+span` (wrapping) whose contacts lie on both faces. When the whole
/// run does, the exact span is kept, so a quarter sheet is not shortened by the sample step.
fn covered_span(
    plane_loc: &Located,
    cyl_loc: &Located,
    start: f64,
    span: f64,
    plane_contact: &impl Fn(f64) -> Vec3,
    cyl_contact: &impl Fn(f64) -> Vec3,
    tol: f64,
) -> Result<(f64, f64)> {
    let n = 48;
    let mut inside = [false; 49];
    for i in 0..=n {
        let th = start + span * (i as f64) / (n as f64);
        inside[i] = on_face(plane_loc, plane_contact(th), tol) && on_face(cyl_loc, cyl_contact(th), tol);
    }
    if inside.iter().all(|x| *x) {
        return Ok((start, span));
    }
    // The longest contiguous run. A wrap joins the ends.
    let mut best: Option<(usize, usize)> = None;
    let mut i = 0;
    while i <= n {
        if !inside[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j <= n && inside[j] {
            j += 1;
        }
        if best.is_none_or(|(a, b)| j - i > b - a) {
            best = Some((i, j - 1));
        }
        i = j;
    }
    let (i0, i1) = best.ok_or_else(|| invalid("the faces don't meet or cross"))?;
    let a0 = start + span * (i0 as f64) / (n as f64);
    let a1 = start + span * (i1 as f64) / (n as f64);
    if a1 - a0 < 1e-4 {
        return Err(invalid("the faces don't meet or cross"));
    }
    Ok((a0, a1 - a0))
}

/// `(start, span)` of the shortest arc covering every sample, or a full turn when the samples
/// leave no real gap. Angles are radians, `span` is in `(0, 2π]`.
fn angle_span(samples: &[f64]) -> Option<(f64, f64)> {
    const TAU: f64 = std::f64::consts::TAU;
    let mut a: Vec<f64> = samples
        .iter()
        .filter(|t| t.is_finite())
        .map(|t| {
            let mut x = t.rem_euclid(TAU);
            if x >= TAU - 1e-12 {
                x = 0.0;
            }
            x
        })
        .collect();
    if a.is_empty() {
        return None;
    }
    a.sort_by(|u, v| u.total_cmp(v));
    a.dedup_by(|u, v| (*u - *v).abs() < 1e-8);
    if a.len() == 1 {
        return Some((a[0], 0.0));
    }
    let mut gap_i = 0;
    let mut gap = -1.0;
    for i in 0..a.len() {
        let next = if i + 1 == a.len() { a[0] + TAU } else { a[i + 1] };
        let g = next - a[i];
        if g > gap {
            gap = g;
            gap_i = i;
        }
    }
    let step = TAU / a.len() as f64;
    if gap < step * 2.5 {
        return Some((0.0, TAU));
    }
    let start = if gap_i + 1 == a.len() { a[0] } else { a[gap_i + 1] };
    Some((start, TAU - gap))
}

/// The quarter (or so) of the ball between the two contacts, through the point halfway in angle.
fn tube_arc(center: Vec3, plane_at: Vec3, cyl_at: Vec3, r: f64) -> Result<mt::Curve> {
    let d0 = plane_at - center;
    let d1 = cyl_at - center;
    let mid = unit(d0 + d1, "the fillet is too flat to build")? * r;
    let edge = mt::builder::circle_arc(&mt::builder::vertex(p3(plane_at)), &mt::builder::vertex(p3(cyl_at)), p3(center + mid));
    Ok(edge.oriented_curve())
}

fn revolved_line(origin: Vec3, axis: Vec3, p0: Vec3, p1: Vec3, sweep: f64) -> Result<Body> {
    revolved_profile(mt::Curve::Line(mt::Line(p3(p0), p3(p1))), origin, axis, sweep)
}

fn revolved_profile(profile: mt::Curve, origin: Vec3, axis: Vec3, sweep: f64) -> Result<Body> {
    const TAU: f64 = std::f64::consts::TAU;
    let (u0, u1) = profile.range_tuple();
    let rev = mt::RevolutedCurve::by_revolution(profile, p3(origin), v3(axis));
    let s = mt::Surface::RevolutedCurve(mt::Processor::new(rev));
    let faces = if sweep > TAU - 1e-3 {
        vec![face_on(&s, u0, u1, 0.0, std::f64::consts::PI)?, face_on(&s, u0, u1, std::f64::consts::PI, TAU)?]
    } else {
        vec![face_on(&s, u0, u1, 0.0, sweep)?]
    };
    Ok(surface_body(faces))
}

fn face_on(s: &mt::Surface, u0: f64, u1: f64, v0: f64, v1: f64) -> Result<mt::Face> {
    let corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)];
    let verts: Vec<mt::Vertex> = corners.iter().map(|(u, a)| mt::Vertex::new(s.subs(*u, *a))).collect();
    let mut edges = Vec::with_capacity(4);
    for i in 0..4 {
        let ((ua, va), (ub, vb)) = (corners[i], corners[(i + 1) % 4]);
        let c = if (va - vb).abs() <= 1e-12 {
            crate::face_params::iso_curve(s, true, va, ua, ub)?
        } else {
            crate::face_params::iso_curve(s, false, ua, va, vb)?
        };
        let (Some(v_from), Some(v_to)) = (verts.get(i), verts.get((i + 1) % 4)) else {
            return Err(invalid("a fillet face with no corners"));
        };
        edges.push(mt::Edge::new(v_from, v_to, c));
    }
    Ok(mt::Face::new(vec![edges.into()], s.clone()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{face_surface, measure, patch_region};
    use solidcraft_geom::{Loop2, Plane, Region2, Vec2};
    use std::f64::consts::{PI, TAU};

    fn rel(a: f64, b: f64) -> f64 {
        (a - b).abs() / b.abs().max(1e-9)
    }
    fn area(b: &Body) -> f64 {
        measure(b).unwrap().area
    }
    fn sheet(plane: Plane, u0: f64, v0: f64, u1: f64, v1: f64) -> Body {
        let (a, b) = (u0.min(u1), u0.max(u1));
        let (c, d) = (v0.min(v1), v0.max(v1));
        let r = Region2 {
            outer: Loop2::polygon(&[Vec2::new(a, c), Vec2::new(b, c), Vec2::new(b, d), Vec2::new(a, d)]),
            holes: vec![],
        };
        patch_region(&plane, &r).unwrap()
    }
    /// A cylindrical sheet: the line `p0`–`p1` revolved about `axis` through `origin` by `sweep`.
    fn strip(origin: Vec3, axis: Vec3, p0: Vec3, p1: Vec3, sweep: f64) -> Body {
        revolved_line(origin, axis, p0, p1, sweep).unwrap()
    }

    #[test]
    fn perpendicular_planes_make_a_quarter_cylinder() {
        let floor = sheet(Plane::XY, 0.0, 0.0, 40.0, 20.0);
        // XZ: sketch y = −Z, so v = −20 is z = +20. Normal +Y.
        let wall = sheet(Plane::XZ, 0.0, 0.0, 40.0, -20.0);
        let pa = Vec3::new(20.0, 10.0, 0.0);
        let pb = Vec3::new(20.0, 0.0, 10.0);
        let blend = surface_fillet(&floor, pa, &wall, pb, 5.0, Trim::None).unwrap();
        let want = 5.0 * (PI / 2.0) * 40.0;
        assert!(rel(area(&blend), want) < 1e-3, "{} vs {want}", area(&blend));
        let FaceSurface::Cylinder { radius, .. } = face_surface(&blend, 0).unwrap() else { panic!("not a cylinder") };
        assert!((radius - 5.0).abs() < 1e-6, "{radius}");
        let sewn = surface_fillet(&floor, pa, &wall, pb, 5.0, Trim::Both).unwrap();
        let want_all = 600.0 + 600.0 + want;
        assert!(rel(area(&sewn), want_all) < 1e-3, "{} vs {want_all}", area(&sewn));
        assert_eq!(sewn.face_count(), 3);
    }

    #[test]
    fn crossing_picks_put_the_spine_in_opposite_quadrants() {
        let floor = sheet(Plane::XY, -20.0, -20.0, 20.0, 20.0);
        let wall = sheet(Plane::XZ, -20.0, -20.0, 20.0, 20.0);
        let pos = surface_fillet(&floor, Vec3::new(0.0, 10.0, 0.0), &wall, Vec3::new(0.0, 0.0, 10.0), 5.0, Trim::None).unwrap();
        let neg = surface_fillet(&floor, Vec3::new(0.0, -10.0, 0.0), &wall, Vec3::new(0.0, 0.0, -10.0), 5.0, Trim::None).unwrap();
        let cp = measure(&pos).unwrap().centroid;
        let cn = measure(&neg).unwrap().centroid;
        assert!(cp.y > 0.0 && cp.z > 0.0, "{cp:?}");
        assert!(cn.y < 0.0 && cn.z < 0.0, "{cn:?}");
    }

    #[test]
    fn parallel_planes_and_a_ball_that_cannot_touch_are_refused() {
        let a = sheet(Plane::XY, 0.0, 0.0, 20.0, 20.0);
        let b = sheet(Plane::XY.offset(8.0), 0.0, 0.0, 20.0, 20.0);
        let err = surface_fillet(&a, Vec3::new(10.0, 10.0, 0.0), &b, Vec3::new(10.0, 10.0, 8.0), 2.0, Trim::None).unwrap_err();
        assert!(err.to_string().contains("the ball can't touch both"), "{err}");
        // A point off the face, and a zero radius, also refuse without panicking.
        let off = surface_fillet(&a, Vec3::new(10.0, 10.0, 3.0), &b, Vec3::new(10.0, 10.0, 8.0), 2.0, Trim::None).unwrap_err();
        assert!(off.to_string().contains("not on a face"), "{off}");
        assert!(surface_fillet(&a, Vec3::new(10.0, 10.0, 0.0), &b, Vec3::new(10.0, 10.0, 8.0), 0.0, Trim::None).is_err());
    }

    #[test]
    fn a_plane_and_a_parallel_cylinder_make_a_cylinder_of_the_ball_radius() {
        let plane = sheet(Plane::XY, -5.0, 0.0, 40.0, 30.0);
        // Axis +Y would turn the bottom toward −X. −Y turns it toward +X, across the contact.
        let cyl = strip(Vec3::new(0.0, 0.0, 10.0), -Vec3::Y, Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 30.0, 0.0), PI / 2.0);
        let r = 5.0;
        let blend = surface_fillet(&plane, Vec3::new(20.0, 15.0, 0.0), &cyl, Vec3::new(10.0, 15.0, 10.0), r, Trim::None).unwrap();
        let x = (40.0 * r).sqrt();
        let radial_u = Vec3::new(x, 0.0, r - 10.0).normalized().unwrap();
        let to_plane = Vec3::new(0.0, 0.0, -1.0);
        let to_cyl = -radial_u;
        let ang = Vec3::Y.dot(to_plane.cross(to_cyl)).atan2(to_plane.dot(to_cyl)).abs();
        let want = r * ang * 30.0;
        assert!(rel(area(&blend), want) < 1e-3, "{} vs {want}", area(&blend));
        let FaceSurface::Cylinder { radius, .. } = face_surface(&blend, 0).unwrap() else { panic!("not a cylinder") };
        assert!((radius - r).abs() < 1e-5, "{radius}");
    }

    #[test]
    fn a_plane_and_a_square_cylinder_make_a_torus_quarter() {
        let plane = sheet(Plane::XY, -5.0, -5.0, 30.0, 30.0);
        let cyl = strip(Vec3::ZERO, Vec3::Z, Vec3::new(10.0, 0.0, 0.0), Vec3::new(10.0, 0.0, 20.0), PI / 2.0);
        let (r, major) = (2.0, 12.0);
        let blend = surface_fillet(&plane, Vec3::new(20.0, 5.0, 0.0), &cyl, Vec3::new(10.0, 0.0, 10.0), r, Trim::None).unwrap();
        // Pappus on the quarter tube: arc r·π/2, centroid closer to the axis by 2r/π along the bisector.
        let want = r * (PI / 2.0) * (major * PI / 2.0 - r);
        assert!(rel(area(&blend), want) < 1e-2, "{} vs {want}", area(&blend));
        assert!(surface_fillet(&plane, Vec3::new(20.0, 5.0, 0.0), &cyl, Vec3::new(10.0, 0.0, 10.0), r, Trim::Both).is_err());
        let _ = TAU;
    }
}
