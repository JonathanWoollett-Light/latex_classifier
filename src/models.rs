use num::FromPrimitive;
use std::ops::{Add, Div, Sub, SubAssign};

// Struct for 2d bound

#[repr(C)]
#[derive(Clone, Debug)]
pub struct Bound<T: Ord + Copy> {
    pub min: Point<T>,
    pub max: Point<T>,
}
impl<T: Ord + Sub<Output = T> + Copy> SubAssign<Point<T>> for Bound<T> {
    fn sub_assign(&mut self, other: Point<T>) {
        *self = Self {
            min: self.min - other,
            max: self.max - other,
        }
    }
}
impl<T: Ord + Add<Output = T> + Div<Output = T> + Copy + FromPrimitive> Bound<T> {
    pub fn i_center(&self) -> T {
        (self.min.i + self.max.i) / FromPrimitive::from_usize(2usize).unwrap()
    }
    // If x bounds of self contain all x bounds in `others`.
    pub fn contains_j(&self, others: &[&Bound<T>]) -> bool {
        for bound in others {
            if bound.min.i < self.min.i || bound.max.i > self.max.i {
                return false;
            }
        }
        true
    }
}
// Constructs bound around given bounds
impl<T: Ord + Copy> From<&Vec<&Bound<T>>> for Bound<T> {
    fn from(bounds: &Vec<&Bound<T>>) -> Self {
        Bound {
            min: Point::new(
                bounds.iter().min_by_key(|p| p.min.i).unwrap().min.i,
                bounds.iter().min_by_key(|p| p.min.j).unwrap().min.j
            ),
            max: Point::new(
                bounds.iter().max_by_key(|p| p.max.i).unwrap().max.i,
                bounds.iter().max_by_key(|p| p.max.j).unwrap().max.j,
            ),
        }
    }
}
impl<T: Ord + Copy> From<(T,T,T,T)> for Bound<T> {
    fn from(b: (T,T,T,T)) -> Self {
        Bound {
            min: Point::new(b.0,b.1),
            max: Point::new(b.2,b.3)
        }
    }
}

// Struct 2p point coordinates
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point<T: Ord> {
    pub i: T,
    pub j: T,
}
impl<T:Ord> Point<T> {
    pub fn new(i:T,j:T) -> Self {
        Point { i, j }
    }
}
impl<T: Ord + Sub<Output = T>> Sub for Point<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            i: self.i - other.i,
            j: self.j - other.j,
        }
    }
}
impl<T: Ord + Sub<Output = T> + Copy> SubAssign for Point<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            i: self.i - other.i,
            j: self.j - other.j,
        }
    }
}

#[derive(Debug)]
pub struct Line {
    pub max: usize,
    pub min: usize,
}

#[derive(Clone, Eq, PartialEq,Debug)]
pub enum BinaryPixel {
    White,
    Black,
    Assigned,
}

#[repr(C)]
#[derive(Debug)]
pub struct BinarizationParameters {
    pub extreme_boundary: u8,
    pub global_boundary: u8,
    pub local_boundary: u8,
    pub field_reach: usize,
    pub field_size: usize,
}

#[repr(C)]
#[derive(Debug)]
pub struct CArray<T> {
    pub ptr: *mut T,
    pub size: usize,
}
impl<T> CArray<T> {
    pub fn new(v: Vec<T>) -> Self {
        let (ptr, size, _) = v.into_raw_parts();
        CArray {
            ptr: ptr,
            size: size,
        }
    }
}
impl From<Vec<Vec<Pixel>>> for CArray<u8> {
    fn from(img: Vec<Vec<Pixel>>) -> Self {
        let vec: Vec<u8> = img
            .into_iter()
            .flatten()
            .flat_map(|p| vec![p.r, p.g, p.b])
            .collect();
        let (ptr, size, _) = vec.into_raw_parts();
        CArray {
            ptr: ptr,
            size: size,
        }
    }
}

#[repr(C)]
#[derive(Debug)]
pub struct SymbolPixels {
    pub pixels: CArray<u8>,
    pub bound: Bound<u32>,
}
impl SymbolPixels {
    pub fn new(pixels: Vec<u8>, bound: Bound<u32>) -> Self {
        SymbolPixels {
            pixels: CArray::new(pixels),
            bound,
        }
    }
}

pub struct Pixel {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub luma: u8,
}
impl Pixel {
    pub fn colour(&mut self, buf: [u8; 3]) {
        self.r = buf[0];
        self.b = buf[1];
        self.g = buf[2];
    }
}
impl From<&[u8]> for Pixel {
    fn from(buf: &[u8]) -> Pixel {
        Pixel {
            r: buf[0],
            g: buf[1],
            b: buf[2],
            luma: buf[0] / 3 + buf[1] / 3 + buf[2] / 3,
        }
    }
}
impl From<Pixel> for Vec<u8> {
    fn from(pixel: Pixel) -> Vec<u8> {
        vec![pixel.r, pixel.g, pixel.b]
    }
}
impl From<&Pixel> for Vec<u8> {
    fn from(pixel: &Pixel) -> Vec<u8> {
        vec![pixel.r, pixel.g, pixel.b]
    }
}

pub struct CReturn {
    pub symbols: CArray<CArray<SymbolPixels>>,
    pub pixels: CArray<u8>
}