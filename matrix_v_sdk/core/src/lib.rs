use dashmap::DashMap;
use fxhash::FxBuildHasher;
use std::os::raw::c_float;
use std::sync::Arc;

// Tile structure: flat vector of floats with dimensions
#[derive(Clone, PartialEq, Eq, Hash)]
struct Tile {
    data: Vec<i64>, // Using fixed-point i64 for exact bit-matching/hashing (9 decimal places)
    rows: usize,
    cols: usize,
}

// Global concurrent cache
// Maps (TileA, TileB) -> TileC
type CacheKey = (Tile, Tile);
type TileCache = DashMap<CacheKey, Tile, FxBuildHasher>;

lazy_static::lazy_static! {
    static ref GLOBAL_CACHE: Arc<TileCache> = Arc::new(DashMap::with_hasher(FxBuildHasher::default()));
}

#[no_mangle]
pub extern "C" fn vl_cache_check(
    a_ptr: *const c_float,
    a_rows: usize,
    a_cols: usize,
    b_ptr: *const c_float,
    b_rows: usize,
    b_cols: usize,
    out_ptr: *mut c_float,
) -> i32 {
    let tile_a = unsafe {
        let slice = std::slice::from_raw_parts(a_ptr, a_rows * a_cols);
        Tile {
            data: slice
                .iter()
                .map(|&x| (x as f64 * 1_000_000_000.0) as i64)
                .collect(),
            rows: a_rows,
            cols: a_cols,
        }
    };

    let tile_b = unsafe {
        let slice = std::slice::from_raw_parts(b_ptr, b_rows * b_cols);
        Tile {
            data: slice
                .iter()
                .map(|&x| (x as f64 * 1_000_000_000.0) as i64)
                .collect(),
            rows: b_rows,
            cols: b_cols,
        }
    };

    let key = (tile_a, tile_b);
    if let Some(res) = GLOBAL_CACHE.get(&key) {
        unsafe {
            let out_slice = std::slice::from_raw_parts_mut(out_ptr, res.rows * res.cols);
            for (i, &val) in res.data.iter().enumerate() {
                out_slice[i] = (val as f64 / 1_000_000_000.0) as f32;
            }
        }
        return 1; // Cache Hit
    }

    0 // Cache Miss
}

#[no_mangle]
pub extern "C" fn vl_cache_insert(
    a_ptr: *const c_float,
    a_rows: usize,
    a_cols: usize,
    b_ptr: *const c_float,
    b_rows: usize,
    b_cols: usize,
    res_ptr: *const c_float,
    res_rows: usize,
    res_cols: usize,
) {
    let tile_a = unsafe {
        let slice = std::slice::from_raw_parts(a_ptr, a_rows * a_cols);
        Tile {
            data: slice
                .iter()
                .map(|&x| (x as f64 * 1_000_000_000.0) as i64)
                .collect(),
            rows: a_rows,
            cols: a_cols,
        }
    };

    let tile_b = unsafe {
        let slice = std::slice::from_raw_parts(b_ptr, b_rows * b_cols);
        Tile {
            data: slice
                .iter()
                .map(|&x| (x as f64 * 1_000_000_000.0) as i64)
                .collect(),
            rows: b_rows,
            cols: b_cols,
        }
    };

    let tile_res = unsafe {
        let slice = std::slice::from_raw_parts(res_ptr, res_rows * res_cols);
        Tile {
            data: slice
                .iter()
                .map(|&x| (x as f64 * 1_000_000_000.0) as i64)
                .collect(),
            rows: res_rows,
            cols: res_cols,
        }
    };

    GLOBAL_CACHE.insert((tile_a, tile_b), tile_res);
}

#[no_mangle]
pub extern "C" fn vl_cache_clear() {
    GLOBAL_CACHE.clear();
}
