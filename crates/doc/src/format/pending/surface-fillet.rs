//! `SurfaceFillet` (Surface Fillet, #769) saves two faces (`a` and `b`, each a `body` and a
//! `point` that picks the side), a `radius`, and `trim` (`both`, `first` or `none`; absent means
//! both). Older designs have no such feature, so they need no transformation. The last release
//! asks for an update instead of building a fillet it cannot read.
