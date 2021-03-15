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
    pub fn y_center(&self) -> T {
        (self.min.y + self.max.y) / FromPrimitive::from_usize(2usize).unwrap()
    }
    // If x bounds of self contain all x bounds in `others`.
    pub fn contains_x(&self, others: &[&Bound<T>]) -> bool {
        for bound in others {
            if bound.min.x < self.min.x || bound.max.x > self.max.x {
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
            min: Point {
                x: bounds.iter().min_by_key(|p| p.min.x).unwrap().min.x,
                y: bounds.iter().min_by_key(|p| p.min.y).unwrap().min.y,
            },
            max: Point {
                x: bounds.iter().max_by_key(|p| p.max.x).unwrap().max.x,
                y: bounds.iter().max_by_key(|p| p.max.y).unwrap().max.y,
            },
        }
    }
}
impl<T: Ord + Copy> From<((T, T), (T, T))> for Bound<T> {
    fn from(bounds: ((T, T), (T, T))) -> Self {
        Bound {
            min: Point {
                x: (bounds.0).0,
                y: (bounds.0).1,
            },
            max: Point {
                x: (bounds.1).0,
                y: (bounds.1).1,
            },
        }
    }
}

// Struct 2p point coordinates
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Point<T: Ord> {
    pub x: T,
    pub y: T,
}
impl<T: Ord + Sub<Output = T>> Sub for Point<T> {
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}
impl<T: Ord + Sub<Output = T> + Copy> SubAssign for Point<T> {
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

#[derive(Debug)]
pub struct Line {
    pub max: usize,
    pub min: usize,
}

#[derive(Clone, Eq, PartialEq)]
pub enum Pixel {
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
    pub field_size: usize
}

#[repr(C)]
#[derive(Debug)]
pub struct CArray<T> {
    pub ptr: *mut T,
    pub size: usize,
}
impl<T> CArray<T> {
    pub fn new(v: Vec<T>) -> Self {
        let (ptr,size,_) = v.into_raw_parts();
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
    pub bound: Bound<u32>
}
impl SymbolPixels {
    pub fn new(pixels: Vec<u8>, bound: Bound<u32> ) -> Self {
        SymbolPixels { pixels: CArray::new(pixels), bound }
    }
}