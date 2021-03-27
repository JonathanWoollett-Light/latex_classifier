use crate::models::*;

use std::{cmp, collections::VecDeque, path::Path, str, usize};

use image::{imageops::FilterType, ImageBuffer, Luma, Rgb};

#[cfg(debug_assertions)]
use crate::util::time;
#[cfg(debug_assertions)]
use num_format::{Locale, ToFormattedString};
#[cfg(debug_assertions)]
use std::time::Instant;

const SCALE: u32 = 24u32;

// Returns symbol images and bounds
// ~O((nm)^2)
#[no_mangle]
pub extern "C" fn segment_buffer(
    img_raw: *const CArray<u8>,
    j_size: usize,
    i_size: usize,
    bin_parameters: *const BinarizationParameters,
) -> *mut CReturn {
    let img_slice = unsafe { std::slice::from_raw_parts((*img_raw).ptr, (*img_raw).size) };

    let safe_bin_parameters = unsafe { &*bin_parameters };

    let mut img_vec = from_buffer(img_slice, (i_size, j_size));

    #[cfg(debug_assertions)]
    let start = Instant::now();

    // 2d vector of size of image, where each pixel will be labelled white/black (and later used in forest fire)
    // O(2nm + (s+r)^2 * nm/s^2
    let binary_pixels = binarize_buffer(
        &mut img_vec,
        safe_bin_parameters.extreme_boundary,
        safe_bin_parameters.global_boundary,
        safe_bin_parameters.local_boundary,
        safe_bin_parameters.field_reach,
        safe_bin_parameters.field_size,
    );

    #[cfg(debug_assertions)]
    output_luma(&img_vec, "binary_image");

    // Gets lists of pixels belonging to each symbol
    // O(nm)
    let pixel_lists: Vec<Vec<(usize, usize)>> = get_pixel_groups_buffer(binary_pixels);

    #[cfg(debug_assertions)]
    println!("pixel_lists.len(): {}", pixel_lists.len());

    // Gets bounds, square bounds and square bounds scaling property for each symbol
    let bounds: Vec<(Bound<usize>, (Bound<i32>, i32))> = get_bounds(&pixel_lists);
    write_bounds(&bounds, &mut img_vec, [0, 255, 0], Some([255, 0, 0]));

    #[cfg(debug_assertions)]
    output_colour(&img_vec, "border_image");

    // Gets lines in-between bounds
    let lines: Vec<Line> = get_lines(bounds.iter().map(|b| &b.0).collect(), i_size);
    write_lines(&lines, &mut img_vec, [0, 0, 255], Some(20));

    #[cfg(debug_assertions)]
    output_colour(&img_vec, "line_image");
    
    // Gets scaled pixels belonging to each symbol
    let symbols: Vec<Vec<u8>> = get_symbols(&pixel_lists, &bounds);

    // print_symbols(&symbols);

    let symbol_lines: Vec<Vec<(Vec<u8>, Bound<usize>)>> =
        split_symbols_by_lines(symbols, bounds.into_iter().map(|(b, _)| b).collect(), lines);

    // print_lines(&symbol_lines);
    
    #[cfg(debug_assertions)]
    println!("{} : Finished segmentation", time(start));

    let mut container: Vec<CArray<SymbolPixels>> = Vec::new();
    for i in 0..symbol_lines.len() {
        let vec: Vec<SymbolPixels> = symbol_lines[i]
            .iter()
            .map(|(pixels, bound)| {
                // TODO Do this cast better
                let u32_bound = Bound {
                    min: Point::new(bound.min.i as u32,bound.min.j as u32),
                    max: Point::new(bound.max.i as u32,bound.max.j as u32)
                };
                //println!("pixels: {}",pixels.len());
                SymbolPixels::new(pixels.clone(), u32_bound)
            })
            .collect();

        let arr: CArray<SymbolPixels> = CArray::new(vec);
        container.push(arr);
    }

    let rtn_pixels: Vec<u8> = img_vec.into_iter().flatten().flat_map(|p|Vec::from(p)).collect();
    //panic!("got to end");

    let rtn = Box::new(CReturn { symbols: CArray::new(container), pixels: CArray::new(rtn_pixels)  });

    Box::into_raw(rtn)
    //symbol_lines
}

#[allow(dead_code)]
fn output_luma(pixels: &Vec<Vec<Pixel>>, name: &str) {
    let vec: Vec<u8> = pixels.iter().flatten().map(|p| p.luma).collect();
    let binary_image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
        pixels[0].len() as u32,
        pixels.len() as u32,
        vec
    )
    .expect("Image creation failed");
    binary_image
        .save(format!("{}.png", name))
        .expect("Image saving failed");
}
#[allow(dead_code)]
fn output_colour(pixels: &Vec<Vec<Pixel>>, name: &str) {
    let vec: Vec<u8> = pixels.iter().flatten().flat_map(|p| Vec::from(p)).collect();
    let image =
        ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(pixels[0].len() as u32,pixels.len() as u32, vec)
            .expect("Image creation failed");
    image
        .save(format!("{}.png", name))
        .expect("Image saving failed");
}
#[allow(dead_code)]
fn print_lines(lines: &Vec<Vec<(Vec<u8>, Bound<usize>)>>) {
    println!("---------------------------------------------");
    println!("Lines:");
    println!("---------------------------------------------");
    for l in lines.iter() {
        println!("Line:");
        for s in l.iter() {
            println!();
            //let ndarray_arr = Array::from_shape_vec((24,24),line.to_vec()).expect("ndarray error");
            print_symbol(&s.0);
        }
    }
    println!("---------------------------------------------");
}
#[allow(dead_code)]
fn print_symbols(symbols: &Vec<Vec<u8>>) {
    println!("---------------------------------------------");
    println!("Symbols:");
    println!("---------------------------------------------");
    for s in symbols.iter() {
        println!();
        print_symbol(s);
    }
    println!("---------------------------------------------");
}
#[allow(dead_code)]
fn print_symbol(symbol: &Vec<u8>) {
    for y in 0..24 {
        for x in 0..24 {
            //print!("{} ",s[y*24+x]);
            if symbol[y*24+x] == 1 { print!("-"); } else { print!("*"); }
        }
        println!();
    }
}

// Returns symbol images and bounds
// ~O((nm)^2)
#[no_mangle]
pub extern "C" fn segment_file(
    path_arr: *const CArray<u8>,
    bin_parameters: *const BinarizationParameters,
) -> *mut CReturn {
    // TODO Why does this cause CFFI to break?
    // let path = unsafe { str::from_raw_parts((*path_arr).ptr, (*path_arr).size,(*path_arr).size) };

    let path_buf: &[u8] = unsafe { std::slice::from_raw_parts((*path_arr).ptr, (*path_arr).size) };
    let path = str::from_utf8(path_buf).expect("string fail");

    // Open the image to segment (in this case it will reside within the `test_imgs` directory)
    let img = image::open(Path::new(&path))
        .expect("Opening image failed")
        .to_rgb8();
    let dims = img.dimensions();
    // panic!("dims: {:.?}",dims);

    // Gets raw pixel values from the image.
    let img_raw: Vec<u8> = img.into_raw();

    println!("img_raw.len(): {}",img_raw.len());

    let black: Vec<usize> = img_raw.iter().enumerate().filter_map(|(i,p)| if *p==0 { Some(i) } else { None }).collect();
    print!("[ ");
    for i in 0..10 {
        print!("{} ",black[i]);
    }
    println!("]");

    // panic!("dims:");
    let box_img = Box::new(CArray::new(img_raw));

    segment_buffer(
        Box::into_raw(box_img),
        dims.0 as usize,
        dims.1 as usize,
        bin_parameters,
    )
}

fn from_buffer(raw: &[u8], (_i_size, j_size): (usize, usize)) -> Vec<Vec<Pixel>> {
    // panic!("raw.len:{}",raw.len());
    let j_size_3 = 3 * j_size;
    // panic!("{}:{}",raw.len(), j_size * i_size_3);
    // raw.len() == j_size * i_size_3
    (0..raw.len())
        .step_by(j_size_3)
        .map(|index| {
            raw[index..index + j_size_3]
                .chunks_exact(3)
                .map(|c| Pixel::from(c))
                .collect::<Vec<Pixel>>()
        })
        .collect()
}

// O(2nm + (s+r)^2 * (n/s)(m/s))
// s = local field size
// r = local field reach
pub fn binarize_buffer(
    img: &mut Vec<Vec<Pixel>>,

    extreme_luma_boundary: u8,
    global_luma_boundary: u8,
    local_luma_boundary: u8,
    local_field_reach: usize,
    local_field_size: usize,
) -> Vec<Vec<BinaryPixel>> {
    #[cfg(debug_assertions)]
    let start = Instant::now();

    let (i_size, j_size) = (img.len(), img[0].len());

    #[cfg(debug_assertions)]
    println!(
        "{} * {} = {}",
        i_size.to_formatted_string(&Locale::en),
        j_size.to_formatted_string(&Locale::en),
        img.len().to_formatted_string(&Locale::en)
    );

    // Gets average luma among pixels
    //  Uses `.fold()` instead of `.sum()` as sum of values will likely exceed `u8:MAX`
    // O(nm)
    let global_luma: u8 =
        (img.iter().flatten().map(|p| p.luma as u32).sum::<u32>() / (i_size * j_size) as u32) as u8;
    println!("global_luma: {}",global_luma);
    let global_luma_sub = global_luma.checked_sub(global_luma_boundary);
    let global_luma_add = global_luma.checked_add(global_luma_boundary);

    let i_chunks = (i_size as f32 / local_field_size as f32).ceil() as usize;
    let j_chunks = (j_size as f32 / local_field_size as f32).ceil() as usize;

    // #[cfg(debug_assertions)]
    println!("{}:{}", i_chunks, j_chunks);
    // panic!("got here {}",chunks);

    let mut local_luma: Vec<Vec<u8>> = vec![vec![u8::default(); j_chunks]; i_chunks];

    // O((s+r)^2*(n/s)*(m/s))
    for (i_chunk, i) in (0..i_size).step_by(local_field_size).enumerate() {
        let i_range = zero_checked_sub(i, local_field_reach)
            ..cmp::min(i + local_field_size + local_field_reach, i_size);
        let i_range_length = i_range.end - i_range.start;
        for (j_chunk, j) in (0..j_size).step_by(local_field_size).enumerate() {
            let j_range = zero_checked_sub(j, local_field_reach)
                ..cmp::min(j + local_field_size + local_field_reach, j_size);
            let j_range_length = j_range.end - j_range.start;

            let total: u32 = i_range
                .clone()
                .map(|i_range_indx| {
                    img[i_range_indx][j_range.clone()]
                        .iter()
                        .map(|p| p.luma as u32)
                        .sum::<u32>()
                })
                .sum();

            // print!(" total:{}",total);
            // println!("{},{} \n {:.?},{:.?} \n {},{}",i_size,x_size,i_r,x_r,i,x);
            // if (i_size * x_size) == 0 || total / (i_size * x_size) as u16 > u8::max_value() as u16 {
            //     println!("{},{} \n {:.?},{:.?} \n {},{}",i_size,x_size,i_r,x_r,i,x);
            // }
            // if i_chunk==0 && j_chunk == 0 {
            //     println!("{} / {} = {}",total,(i_range_length*j_range_length),(total / (i_range_length*j_range_length) as u32) as u8);
            // }
            local_luma[i_chunk][j_chunk] = (total / (i_range_length*j_range_length) as u32) as u8;
            // println!(" local_luma[i]:{}",local_luma[i]);
        }
    }
    // println!("local_luma: {}",local_luma[0][0]);

    #[cfg(debug_assertions)]
    println!("{} : Calculated local luminance fields", time(start));

    #[cfg(debug_assertions)]
    let start = Instant::now();

    // Initializes 2d vector of size i_size*j_size with BinariPixel::White.
    let mut pixels: Vec<Vec<BinaryPixel>> = vec![vec!(BinaryPixel::White; j_size); i_size];

    //println!("{}:{}->{}:({},{})",j_size,i_size,global_luma,local_luma.len(),local_luma[0].len());
    // let mut count_1 = 0;
    // let mut count_2 = 0;
    // let mut count_3 = 0;
    // let mut count_4 = 0;
    // let mut count_5 = 0;
    // let mut count_6 = 0;
    
    // O(nm)
    for i in 0..i_size {
        let i_group: usize = i / local_field_size; // == floor(i as f32 / local_field_size as f32) as usize
        for j in 0..j_size {
            let j_group: usize = j / local_field_size;

            // Extreme global boundaries
            // --------------------------------
            if img[i][j].luma < extreme_luma_boundary {
                img[i][j].luma = 0;
                pixels[i][j] = BinaryPixel::Black;
                // count_1 += 1;
                continue;
            }
            if img[i][j].luma > 255 - extreme_luma_boundary {
                img[i][j].luma = 255;
                // count_2 += 1;
                // pixels `BinaryPixel::White` bi default
                continue;
            }

            // Global average boundaries
            // --------------------------------
            if let Some(global_lower) = global_luma_sub {
                if img[i][j].luma < global_lower {
                    img[i][j].luma = 0;
                    pixels[i][j] = BinaryPixel::Black;
                    // count_3 += 1;
                    continue;
                }
            }
            if let Some(global_upper) = global_luma_add {
                if img[i][j].luma > global_upper {
                    img[i][j].luma = 255;
                    // count_4 += 1;
                    // pixels `BinaryPixel::White` bi default
                    continue;
                }
            }

            // Local average boundaries
            // --------------------------------
            if let Some(local) = local_luma[i_group][j_group].checked_sub(local_luma_boundary) {
                if img[i][j].luma < local {
                    img[i][j].luma = 0;
                    pixels[i][j] = BinaryPixel::Black;
                    // count_5 += 1;
                    continue;
                }
            }
            if let Some(local) = local_luma[i_group][j_group].checked_add(local_luma_boundary) {
                if img[i][j].luma > local {
                    img[i][j].luma = 255;
                    // pixels `BinaryPixel::White` bi default
                    // count_6 += 1;
                    continue;
                }
            }
            // White is the negative (false/0) colour in our binarization, thus this is our else case
            img[i][j].luma = 255;
            // pixels `BinaryPixel::White` bi default
        }
    }
    // println!("{}   {}   {}   {}   {}   {}",count_1,count_2,count_3,count_4,count_5,count_6);
    // panic!("got here");

    #[cfg(debug_assertions)]
    println!("{} : Converted image to binary", time(start));

    pixels
}

fn zero_checked_sub(a: usize, b: usize) -> usize {
    if let Some(val) = a.checked_sub(b) {
        return val;
    }
    0
}

// O((nm)^2)
pub fn get_pixel_groups_buffer(mut pixels: Vec<Vec<BinaryPixel>>) -> Vec<Vec<(usize, usize)>> {
    #[cfg(debug_assertions)]
    let start = Instant::now();

    // List of lists of pixels belonging to each symbol.
    let mut pixel_lists: Vec<Vec<(usize, usize)>> = Vec::new();

    let (i_size, j_size) = (pixels.len(), pixels[0].len());
    // Iterates through pixels
    for i in 0..i_size {
        for j in 0..j_size {
            if pixels[i][j] == BinaryPixel::Black {
                // Pushes new list to hold pixels belonging to this newly found symbol
                pixel_lists.push(Vec::new());

                // Triggers the forest fire algorithm
                let last_index = pixel_lists.len() - 1;
                forest_fire(
                    &mut pixels,
                    (i_size, j_size),
                    (i, j),
                    &mut pixel_lists[last_index],
                );
            }
        }
    }

    #[cfg(debug_assertions)]
    println!("{} : Fill finished", time(start));

    // let min = j_size * i_size / 10000;
    // let pixel_lists: Vec<Vec<(usize, usize)>> = pixel_lists.into_iter().filter(|pl|pl.len() > min).collect();

    // Filter out shitty specs & return
    let min = i_size * j_size / 20000; // TODO Adapt to use size relative to largest characters
    return pixel_lists
        .into_iter()
        .filter(|pl| pl.len() > min)
        .collect();

    fn forest_fire(
        pixels: &mut Vec<Vec<BinaryPixel>>,
        (i_size, j_size): (usize, usize),
        (i, j): (usize, usize),
        pixel_list: &mut Vec<(usize, usize)>,
    ) {
        // Push 1st pixel to symbol
        pixel_list.push((i,j));
        // Sets 1st pixels symbol number
        pixels[i][j] = BinaryPixel::Assigned;
        // Initialises queue for forest fire
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();
        // Pushes 1st pixel to queue
        queue.push_back((i,j));

        // While there is value in queue (effectively while !queue.is_empty())
        while let Some((i,j)) = queue.pop_front() {
            if i > 0 {
                pixels[i-1][j] = run_pixel((i-1, j), pixels, pixel_list, &mut queue);
            }
            if i < i_size - 1 {
                pixels[i+1][j] = run_pixel((i+1, j), pixels, pixel_list, &mut queue);
            }
            
            if j > 0 {
                pixels[i][j-1] = run_pixel((i, j-1), pixels, pixel_list, &mut queue);
            }
            if j < j_size - 1 {
                pixels[i][j+1] = run_pixel((i, j+1), pixels, pixel_list, &mut queue);
            }
        }
        fn run_pixel(
            (i, j): (usize, usize),
            pixels: &Vec<Vec<BinaryPixel>>,
            pixel_list: &mut Vec<(usize, usize)>,
            queue: &mut VecDeque<(usize, usize)>,
        ) -> BinaryPixel {
            if pixels[i][j] == BinaryPixel::Black {
                // If black pixel unassigned to symbol
                pixel_list.push((i, j)); // Push pixel to symbol
                queue.push_back((i, j)); // Enqueue pixel for forest fire algorithm
                return BinaryPixel::Assigned; // Return value to set `symbols[x][y]` to
            }
            return BinaryPixel::White; // Return value to set `symbols[x][y]` to
        }
    }
}

fn write_bounds(
    bounds: &Vec<(Bound<usize>, (Bound<i32>, i32))>,
    pixels: &mut Vec<Vec<Pixel>>,
    colour: [u8; 3],
    sqr: Option<[u8; 3]>,
) {
    let (i_size, j_size) = (pixels.len(), pixels[0].len());

    for (bound, (sqr_bound, _)) in bounds.iter() {
        let j_range = bound.min.j..cmp::min(bound.max.j + 1, j_size);
        // Lower i
        if bound.min.i > 0 {
            for j in j_range.clone() {
                pixels[bound.min.i - 1][j].colour(colour);
            }
        }
        // Upper i
        if bound.max.i < i_size - 1 {
            for j in j_range.clone() {
                pixels[bound.max.i + 1][j].colour(colour);
            }
        }

        let i_range = bound.min.i..cmp::min(bound.max.i + 1, i_size);

        // Lower j
        if bound.min.j > 0 {
            for i in i_range.clone() {
                pixels[i][bound.min.j - 1].colour(colour);
            }
        }
        // Upper j
        if bound.max.j < j_size - 1 {
            for i in i_range.clone() {
                pixels[i][bound.max.j + 1].colour(colour);
            }
        }

        // Handle square bounds
        if let Some(sqr_colour) = sqr {
            let sqr_i_range =
                cmp::max(sqr_bound.min.i, 0)..cmp::min(sqr_bound.max.i + 1, i_size as i32);

            // Lower j
            if sqr_bound.min.j > 0 {
                for i in sqr_i_range.clone() {
                    pixels[i as usize][sqr_bound.min.j as usize - 1].colour(sqr_colour);
                }
            }
            // Upper j
            if sqr_bound.max.j < j_size as i32 - 1 {
                for i in sqr_i_range.clone() {
                    pixels[i as usize][sqr_bound.max.j as usize + 1].colour(sqr_colour);
                }
            }

            let sqr_j_range =
                cmp::max(sqr_bound.min.j, 0)..cmp::min(sqr_bound.max.j + 1, j_size as i32);

            // Lower i
            if sqr_bound.min.i > 0 {
                for j in sqr_j_range.clone() {
                    pixels[sqr_bound.min.i as usize - 1][j as usize].colour(sqr_colour);
                }
            }
            // Upper i
            if sqr_bound.max.i < i_size as i32 - 1 {
                for j in sqr_j_range.clone() {
                    pixels[sqr_bound.max.i as usize + 1][j as usize].colour(sqr_colour);
                }
            }
        }
    }
}

pub fn write_lines(
    lines: &[Line],
    pixels: &mut Vec<Vec<Pixel>>,
    colour: [u8; 3],
    infill_spacing: Option<usize>,
) {
    #[cfg(debug_assertions)]
    let start = Instant::now();

    let j_size = pixels[0].len();

    // Sets lines
    for line in lines.iter() {
        println!("{}->{} ({})", line.min, line.max, pixels[0].len());
        // Writes boundaries
        for j in 0..j_size {
            pixels[line.min][j].colour(colour);
            pixels[line.max][j].colour(colour);
        }
        // Writes infill
        if let Some(spacing) = infill_spacing {
            for j in 0..j_size {
                for i in line.min..line.max {
                    if (i + j) % spacing == 0 {
                        pixels[i][j].colour(colour);
                    }
                }
            }
        }
    }

    #[cfg(debug_assertions)]
    println!("{} : Lines output", time(start));
}

// TODO This could be compressed into `get_pixel_groups`
// O(nm)
pub fn get_bounds(pixel_lists: &[Vec<(usize, usize)>]) -> Vec<(Bound<usize>, (Bound<i32>, i32))> {
    #[cfg(debug_assertions)]
    let start = Instant::now();
    // Gets bounds
    let mut bounds: Vec<Bound<usize>> = Vec::new();
    let mut sqr_bounds: Vec<(Bound<i32>, i32)> = Vec::new();

    // Iterate across all symbols
    for symbol in pixel_lists {
        let (mut lower_i, mut lower_j) = symbol[0];
        let (mut upper_i, mut upper_j) = symbol[0];
        for p in symbol.iter().skip(1) {
            let &(i, j) = p;

            if i < lower_i {
                lower_i = i;
            } else if i > upper_i {
                upper_i = i;
            }

            if j < lower_j {
                lower_j = j;
            } else if j > upper_j {
                upper_j = j;
            }
        }

        // Gets square bounds centred on original bounds
        let bound = Bound::from((lower_i, lower_j, upper_i, upper_j));
        //println!("bound: {:.?}",bound);
        bounds.push(bound);
        let sqr_bound = square_indxs(lower_i, lower_j, upper_i, upper_j);
        println!("sqr bound: {:.?}",sqr_bound);
        
        sqr_bounds.push(sqr_bound);
    }
    // panic!("are we allowed to panic?");

    #[cfg(debug_assertions)]
    println!("{} : Bounds set", time(start));

    return bounds.into_iter().zip(sqr_bounds.into_iter()).collect();

    // talk about adding this
    fn square_indxs(
        lower_i: usize,
        lower_j: usize,
        upper_i: usize,
        upper_j: usize,
    ) -> (Bound<i32>, i32) {
        #[cfg(debug_assertions)]
        assert!(upper_i > lower_i && upper_j  > lower_j);

        let (i_size, j_size) = (upper_i - lower_i, upper_j - lower_j);
        
        let dif: i32 = i_size as i32 - j_size as i32;
        let dif_2: f32 = dif as f32 / 2.;

        let dif_f = dif_2.floor() as i32;
        let dif_c = dif_2.ceil() as i32;
        let sqr_bound = Bound::from(( 
            cmp::min(lower_i as i32,lower_i as i32 + dif_f),
            cmp::min(lower_j as i32,lower_j as i32 - dif_f),
            cmp::max(upper_i as i32,upper_i as i32 - dif_c),
            cmp::max(upper_j as i32,upper_j as i32 + dif_c)
        ));

        #[cfg(debug_assertions)]
        assert!(sqr_bound.max.i - sqr_bound.min.i == sqr_bound.max.j - sqr_bound.min.j);

        return (sqr_bound,dif_2 as i32);
    }
}

// O(nm/p) => O(p)
// p = min symbol size, right now using nm/20000 therefore O(p) => O(2000) => O(1)
pub fn get_lines(bounds: Vec<&Bound<usize>>, i_size: usize) -> Vec<Line> {
    #[cfg(debug_assertions)]
    let start = Instant::now();

    let mut lines = Vec::with_capacity(bounds.len()); // Number of symbols = maximum number of lines

    if bounds[0].max.i < i_size-1 { 
        lines.push(Line {
            max: i_size-1,
            min: bounds[0].max.i+1,
        });
    }
    if bounds[0].min.i > 0 {
        lines.push(Line {
            max: bounds[0].min.i-1,
            min: 0,
        });
    }
    
    println!("initial lines: {:.?}",lines);
    // Height of tallest symbol
    let mut max_i_size = bounds[0].max.i - bounds[0].min.i;
    // O(nm)
    for b in bounds.iter().skip(1) {
        let h = b.max.i - b.min.i;
        if h > max_i_size {
            max_i_size = h;
        }
        for i in 0..lines.len() {
            // Writing these comparisons slightly more efficiently makes them ugly and hard to understand.
            // So for now, they done this way.

            // Covering (line covering symbol)
            if lines[i].max > b.max.i && lines[i].min < b.min.i {
                lines.push(Line {
                    max: b.min.i-1,
                    min: lines[i].min,
                });
                lines[i].min = b.max.i+1;
            }
            // Upper intersection (lower part of line within symbol)
            else if lines[i].max > b.max.i && lines[i].min <= b.max.i {
                lines[i].min = b.max.i+1;
            }
            // Lower intersection (upper part of line within symbol)
            else if lines[i].max >= b.min.i && lines[i].min < b.min.i {
                lines[i].max = b.min.i-1;
            }
            // Covered (line covered by symbol)
            else if lines[i].max < b.max.i && lines[i].min > b.min.i {
                lines.remove(i);
            }
        }
    }

    #[cfg(debug_assertions)]
    println!("{} : Got lines", time(start));

    // Filter out all small lines (as they could be between super/sub script unless they are edge lines)
    lines
        .into_iter()
        .filter(|l| (l.max - l.min) > (max_i_size / 4) || l.max == i_size-1 || l.min == 0)
        .collect()
}

// Returns tuple of: 2d vec of pixels in symbol, Bounds of symbol
// O(p)
pub fn get_symbols(
    pixel_lists: &[Vec<(usize, usize)>],
    bounds: &[(Bound<usize>, (Bound<i32>, i32))],
) -> Vec<Vec<u8>> {
    #[cfg(debug_assertions)]
    let start = Instant::now();
    let mut symbols: Vec<Vec<u8>> = Vec::with_capacity(pixel_lists.len());

    for (p,b) in pixel_lists.iter().zip(bounds.iter()) {
        // Sets `min_actuals` as minimum bounds and `bounds` as square bounds.
        let (bound, (sqr_bound, sqr_diff)) = &b;

        // Calculates i_size and j_size of image using square bounds
        let i_size: usize = (sqr_bound.max.i - sqr_bound.min.i + 1) as usize;
        let j_size: usize = (sqr_bound.max.j - sqr_bound.min.j + 1) as usize;

        // Constructs list to hold symbol image
        let mut symbol = vec![vec!(255u8; j_size); i_size];

        // Iterates over pixels belonging to symbol
        for &(i, j) in p.iter() {
            // Sets x,y coordinates scaled to square bounds from original x,y coordinates.
            let (scaled_i, scaled_j) = if *sqr_diff < 0 {
                ((i as i32 - sqr_diff) as usize, j)
            } else {
                (i, (j as i32 + sqr_diff) as usize)
            };

            // Sets pixel in symbol image list
            symbol[scaled_i - bound.min.i][scaled_j - bound.min.j] = 0u8;
        }

        // println!("Raw symbol ({}*{}):",i_size,j_size);
        // for i in (0..i_size) {
        //     for j in (0..j_size) {
        //         if symbol[i][j]==255 { print!("-"); } else { print!("*"); }
        //     }
        //     println!();
        // }

        // Constructs image buffer from symbol vector
        let symbol_image = ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(
            i_size as u32,
            j_size as u32,
            symbol.into_iter().flatten().collect(),
        )
        .unwrap();

        // Scales image buffer
        let scaled_image =
            image::imageops::resize(&symbol_image, SCALE, SCALE, FilterType::Triangle);

        // Sets list to image buffer and carries out binarization.
        //  Running basic binarization here is necessary as scaling will have blurred some pixels.
        //  Basic binarization should also be proficient as the blurring will be minor.
        let binary_vec: Vec<u8> = scaled_image
            .into_raw()
            .into_iter()
            .map(|p| if p < 220 { 0 } else { 1 })
            .collect();

        // Pushes the scaled symbol list to the symbols list.
        symbols.push(binary_vec);
    }
    #[cfg(debug_assertions)]
    println!("{} : Symbols set", time(start));

    symbols
}

// O(p log p + p)
fn split_symbols_by_lines(
    mut symbols: Vec<Vec<u8>>,
    mut bounds: Vec<Bound<usize>>,
    mut lines: Vec<Line>,
) -> Vec<Vec<(Vec<u8>, Bound<usize>)>> {
    #[cfg(debug_assertions)]
    let start = Instant::now();

    // Presuming already sorted, might need top come back to this if I'm wrong
    // TODO If this proves to always be true, switch back to stable from nightly
    // O(p log p)
    lines.sort_by_key(|l| l.max);
    #[cfg(debug_assertions)]
    println!("Sorted lines: {:.?}", lines);

    // The `-1` here could cause an error if there was not enough space to create lines at the bottom and top of an image.
    let mut symbol_lines: Vec<Vec<(Vec<u8>, Bound<usize>)>> = vec![Vec::new(); lines.len() - 1];
    // Assuming lines sorted by ascending max (from 0->max, given 0 is top, this is top to bottom technically)
    // O(p)
    for (line, symbol_line) in lines.into_iter().skip(1).zip(symbol_lines.iter_mut()) {
        let mut adjust = 0;
        for i in 0..bounds.len() {
            if bounds[i - adjust].min.i < line.max {
                symbol_line.push((symbols.remove(i - adjust), bounds.remove(i - adjust)));
                adjust += 1;
            }
        }
    }

    #[cfg(debug_assertions)]
    println!(
        "Line j_sizes: {:.?}",
        symbol_lines.iter().map(|s| s.len()).collect::<Vec<usize>>()
    );

    #[cfg(debug_assertions)]
    println!("{} : Split symbols into lines", time(start));

    symbol_lines
}
