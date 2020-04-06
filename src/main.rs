use std::time::{Instant,Duration};

use itertools::izip;
use std::fs::File;
use std::io::Read;
use cogent::core::*;
use image::{ImageBuffer,Luma};
use std::path::Path;
use std::fs;
use std::collections::VecDeque;
use image::imageops::FilterType;
use std::collections::HashMap;
use std::usize;

const LUMA_ADJUSTMENT:u8 = 15u8; // Luma less than set to 0 and more than set to 255.
const B_SPACING:usize = 2usize; // Border space
const WHITE_SPACE_SYMBOL:char = ' '; // What symbol to use when priting white pixels
const ROW_CLEARANCE:f32 = 0.3;
const IMG:&str = "composite.jpg";

fn main() {
    // train_net();
    
    
    let (images,bounds) = segment(true,"composite.jpg");
    // Placeholder for calssification of `images`
    let classes:[char;15] = ['3','2','x','7','1','2','.','b','+','-','y','-','-','-','.'];
    let latex = construct(true,&classes,&bounds);
    println!("latex: {}",latex);

    let (images,bounds) = segment(true,"exponential.jpg");
    // Placeholder for calssification of `images`
    let classes:[char;9] = ['1','2','-','3','4','2','2','x','-'];
    let latex = construct(true,&classes,&bounds);
    println!("latex: {}",latex);

    let (images,bounds) = segment(true,"mix.jpg");
    // Placeholder for calssification of `images`
    let classes:[char;11] = ['3','5','-','4','x','+','2','2','y','x','-'];
    let latex = construct(true,&classes,&bounds);
    println!("latex: {}",latex);

    let (images,bounds) = segment(true,"updown.jpg");
    // Placeholder for calssification of `images`
    let classes:[char;7] = ['1','2','2','x','2','-','1'];
    let latex = construct(true,&classes,&bounds);
    println!("latex: {}",latex);
    
}

fn train_net() {
    //Setup
    let mut data = get_combined("combined_dataset");
    // let k = 92usize;
    // let mut net = NeuralNetwork::new(data[0].0.len(),&[
    //     Layer::new(((data[0].0.len()+k) as f32 / 2f32) as usize,Activation::ReLU),
    //     Layer::new(k,Activation::Softmax)
    // ],None);
    // //Execution
    // net.train(&data)
    //     .learning_rate(0.05)
    //     .tracking()
    //     .log_interval(MeasuredCondition::Iteration(1))
    //     .checkpoint_interval(MeasuredCondition::Iteration(1))
    //     .early_stopping_condition(MeasuredCondition::Iteration(100))
    // .go();
    // neural_network.export("final classifier");
    let mut net = NeuralNetwork::import("checkpoints/24");

    // let mut sorted_data = counting_sort(&data, k);
    // println!("Sorted");

    // let mut last_class = sorted_data[0].1;
    // print!("{}",last_class);
    // for example in sorted_data.iter() {
    //     if example.1 != last_class {
    //         print!("->{}",example.1);
    //         last_class = example.1;
    //     }
    // }
    // println!();

    // compress_classes(&mut sorted_data);
    // println!("classes: {:.?}\n",sorted_data[sorted_data.len()-1].1+1);

    // println!("Compressed");

    // let mut last_class = sorted_data[0].1;
    // print!("{}",last_class);
    // for example in sorted_data.iter() {
    //     if example.1 != last_class {
    //         print!("->{}",example.1);
    //         last_class = example.1;
    //     }
    // }
    // println!();

    let evaluation = net.evaluate_outputs(&mut data);
    //println!("\n{}\n\n",array2_prt(&evaluation.1));

    for i in 0..evaluation.0.len() {
        print!("{} : {:.2} similar: ",i,evaluation.0[i]);
        for t in 0..evaluation.1[i].len() {
            if t == i { continue; }
            if evaluation.1[i][t] > 0.1f32 {
                print!("({}:{:.2}) ",t,evaluation.1[i][t]);
            }
        }
        println!();
    }
    println!();

    // println!("\n{}\n\n",array2_prt(&evaluation.2)); Is far too large to really well analyze


    let evaluation = net.evaluate(&data);
    println!("Cost {}, Accuracy: {}/{} ({}%)",evaluation.0,evaluation.1,data.len(),(100f32 * evaluation.1 as f32 / data.len() as f32) as u8);

    fn get_combined(path:&str) -> Vec<(Vec<f32>,usize)> {
        let mut file = File::open(path).unwrap();
        let mut combined_buffer = Vec::new();
        file.read_to_end(&mut combined_buffer).expect("Couldn't read combined");
    
        let mut combined_vec:Vec<(Vec<f32>,usize)> = Vec::new();
        let multiplier:usize = (35*35)+1;
        let length = combined_buffer.len() / multiplier;
        println!("length: {}",length);
        for i in (0..length).rev() {
            let image_index = i * multiplier;
            let label_index = image_index + multiplier -1;
    
            let label:u8 = combined_buffer.split_off(label_index)[0]; // Array is only 1 element regardless
    
            let image:Vec<u8> = combined_buffer.split_off(image_index);
            let image_f32 = image.iter().map(|&x| x as f32).collect();
    
            combined_vec.push((image_f32,label as usize));
        }
        return combined_vec;
    }
    // Assumes sorted `test_data`
    fn compress_classes(test_data:&mut [(Vec<f32>,usize)]) {
        let mut subtractor = test_data[0].1;
        test_data[0].1 -= subtractor;
        for i in 1..test_data.len() {
            if test_data[i].1 != test_data[i-1].1 + subtractor {
                subtractor = test_data[i].1 - test_data[i-1].1 - 1;
            }
            test_data[i].1 -= subtractor;
        }
    }
}
// Returns symbol images and bounds
fn segment(debug_out:bool,path:&str) -> (Vec<ImageBuffer<Luma<u8>,Vec<u8>>>,Vec<((usize,usize),(usize,usize))>) {
    println!("\n");
    let img = image::open(format!("test_imgs/{}",path)).unwrap().to_luma();
    
    let dims = img.dimensions();
    let (width,height) = (dims.0 as usize,dims.1 as usize);
    
    let mut img_raw:Vec<u8> = img.clone().into_raw();
    
    // 2d vector of size of image, where each pixel will be labelled as to which symbol it belongs
    let mut symbols:Vec<Vec<u32>> = to_bin2dvec(debug_out,width,height,&mut img_raw);

    // Debug stuff to check binarisation worked:
    if debug_out {
        let check_img:ImageBuffer<Luma<u8>,Vec<u8>> = ImageBuffer::<Luma<u8>,Vec<u8>>::from_raw(width as u32,height as u32,img_raw).unwrap();
        if !Path::new("binary_imgs").exists() {
            fs::create_dir("binary_imgs").unwrap();
        }
        check_img.save(format!("binary_imgs/{}.png",path)).unwrap();
    }
    
    // Printing can be nice to visualize process.
    // But for larger images it simply prints useless spam in the console.
    if debug_out && width <= 200 && height <= 400 {
        symbols_intial_prt(&symbols);
    }
    return flood_segmentation(&path,debug_out,width,height,&mut symbols);

    fn flood_segmentation(path:&str,debug_out:bool,width:usize,height:usize,symbols:&mut Vec<Vec<u32>>) -> (Vec<ImageBuffer<Luma<u8>,Vec<u8>>>,Vec<((usize,usize),(usize,usize))>) {
        let start = Instant::now();

        let mut symbol_count = 2u32;
        let mut pixels_in_symbols:Vec<Vec<(usize,usize)>> = Vec::new();

        let start_flood = Instant::now();

        for y in 0..height {
            for x in 0..width {
                if symbols[y][x] == 0 {
                    pixels_in_symbols.push(Vec::new());
                    let last_index = pixels_in_symbols.len()-1;
                    flood_fill_queue(symbols,symbol_count,width,height,x,y,&mut pixels_in_symbols[last_index]);
                    symbol_count += 1;
                }
            }
        }
        if debug_out { println!("{} : Flood finished",time(start_flood)); }
        
        if debug_out && width <= 200 && height <= 400 {
            symbols_classified_prt(&symbols);
        }
    
        // Set borders of symbols
        let (borders,bounds) = set_borders(debug_out,path,symbols,&pixels_in_symbols,width,height);
        // Create symbol images
        let mut symbol_images:Vec<ImageBuffer<Luma<u8>,Vec<u8>>> = create_symbol_images(debug_out,&pixels_in_symbols,&borders,width,height);
        let mut scaled_symbols:Vec<ImageBuffer<Luma<u8>,Vec<u8>>> = Vec::new();
        // Export symbol images

        if debug_out {
            // Create folder
            if !Path::new("split").exists() {
                fs::create_dir("split").unwrap();
            }
            // If folder exists, empty it.
            let path = format!("split/{}",path);
            if Path::new(&path).exists() {
                fs::remove_dir_all(&path).unwrap();// Delete folder
            }
            fs::create_dir(&path).unwrap(); // Create folder
        }
        

        for i in 0..symbol_images.len() {
            let mut scaled_image = image::imageops::resize(&mut symbol_images[i],35,35,FilterType::Triangle);
            image::imageops::colorops::invert(&mut scaled_image);

            if debug_out {
                
                let path = format!("split/{}/{}.png",path,i);
                scaled_image.save(path).unwrap();
            }
            scaled_symbols.push(scaled_image);
        }
        //Export bounds
        if debug_out { println!("{} : Flood segmented",time(start)); }

        return (symbol_images,bounds);
    
        fn flood_fill_queue(symbols:&mut Vec<Vec<u32>>,symbol_count:u32,width:usize,height:usize,x:usize,y:usize,pixels:&mut Vec<(usize,usize)>) {
            pixels.push((x,y));
            symbols[y][x] = symbol_count;
            let mut queue:VecDeque<(usize,usize)> = VecDeque::new();
            queue.push_back((x,y));
            loop {
                if let Some(n) = queue.pop_front() {
                    // +x
                    if n.0 < width-1 {
                        let (x,y) = (n.0+1,n.1);
                        if symbols[y][x] == 0 {
                            pixels.push((x,y));
                            symbols[y][x] = symbol_count;
                            queue.push_back((x,y));
                        }
                    }
                    // -x
                    if n.0 > 0 {
                        let (x,y) = (n.0-1,n.1);
                        if symbols[y][x] == 0 {
                            pixels.push((x,y));
                            symbols[y][x] = symbol_count;
                            queue.push_back((x,y));
                        }
                    }
                    // +y
                    if n.1 < height-1 {
                        let (x,y) = (n.0,n.1+1);
                        if symbols[y][x] == 0 {
                            pixels.push((x,y));
                            symbols[y][x] = symbol_count;
                            queue.push_back((x,y));
                        }
                    }
                    // -y
                    if n.1 > 0 {
                        let (x,y) = (n.0,n.1-1);
                        if symbols[y][x] == 0 {
                            pixels.push((x,y));
                            symbols[y][x] = symbol_count;
                            queue.push_back((x,y));
                        }
                    }
                }
                else { break; } // If queue empty break
            }
        }
        fn set_borders(debug_out:bool,path:&str,symbols:&mut Vec<Vec<u32>>,pixel_symbols:&Vec<Vec<(usize,usize)>>,width:usize,height:usize) -> (Vec<((usize,usize),(usize,usize))>,Vec<((usize,usize),(usize,usize))>) {
            let start = Instant::now();
            // Gets bounds
            let mut border_bounds:Vec<((usize,usize),(usize,usize))> = Vec::new();
            let mut bounds:Vec<((usize,usize),(usize,usize))> = Vec::new();
            for symbol in pixel_symbols {
                let mut lower_x = symbol.iter().fold(width, |min,x| (if x.0 < min { x.0 } else { min }));
                let mut lower_y = symbol.iter().fold(height, |min,x| (if x.1 < min { x.1 } else { min }));
                let mut upper_x = symbol.iter().fold(0usize, |max,x| (if x.0 > max { x.0 } else { max }));
                let mut upper_y = symbol.iter().fold(0usize, |max,x| (if x.1 > max { x.1 } else { max }));
    
                bounds.push(((lower_x,lower_y),(upper_x,upper_y)));
    
                if lower_x >= B_SPACING { lower_x -= B_SPACING; };
                if lower_y >= B_SPACING { lower_y -= B_SPACING; };
                if upper_x + B_SPACING < width { upper_x += B_SPACING; };
                if upper_y + B_SPACING < height { upper_y += B_SPACING; };
    
                border_bounds.push(((lower_x,lower_y),(upper_x,upper_y)));
            }
            // Copies image
            let mut border_img = image::open(format!("test_imgs/{}",path)).unwrap().into_rgb();
            for (x,y,pixel) in border_img.enumerate_pixels_mut() {
                let val = if symbols[y as usize][x as usize] == 1 { 255 } else { 0 };
                *pixel = image::Rgb([val,val,val]);
            }
    
            // Sets borders
            let border_pixel = image::Rgb([255,0,0]); // Pixel to use as border
            for symbol in border_bounds.iter() {
                let min_x = (symbol.0).0;
                let min_y = (symbol.0).1;
                let max_x = (symbol.1).0;
                let max_y = (symbol.1).1;
                // Sets horizontal borders
                for i in min_x..max_x {
                    *border_img.get_pixel_mut(i as u32,min_y as u32) = border_pixel;
                    *border_img.get_pixel_mut(i as u32,max_y as u32) = border_pixel;
                }
                // Sets vertical borders
                for i in min_y..max_y {
                    *border_img.get_pixel_mut(min_x as u32,i as u32) = border_pixel;
                    *border_img.get_pixel_mut(max_x as u32,i as u32) = border_pixel;
                }
                // Sets bottom corner border
                *border_img.get_pixel_mut(max_x as u32,max_y as u32) = border_pixel;
            }
            
            if debug_out {
                if !Path::new("borders").exists() {
                    fs::create_dir("borders").unwrap();
                }
                border_img.save(format!("borders/{}.png",path)).unwrap();
                println!("{} : Borders set",time(start));
            }
            return (border_bounds,bounds);
        }
        fn create_symbol_images(debug_out:bool,pixels_in_symbols:&Vec<Vec<(usize,usize)>>,borders:&Vec<((usize,usize),(usize,usize))>,width:usize,height:usize) -> Vec<ImageBuffer<Luma<u8>,Vec<u8>>> {
            let start = Instant::now();
            let sizes:Vec<(usize,usize)> = borders.iter().map(|lims| ((lims.1).0-(lims.0).0+1,(lims.1).1-(lims.0).1+1)).collect();
            // Default constructs to all black pixels, thus we set symbol pixels to white in following loop
            // TODO Look into constructing with default white pixels and drawing black pixels
            let mut symbol_images:Vec<ImageBuffer<Luma<u8>,Vec<u8>>> = sizes.iter().map(|size| ImageBuffer::<Luma<u8>,Vec<u8>>::new(size.0 as u32,size.1 as u32)).collect();
            // O(n)
            // Draws symbol images
            for i in 0..pixels_in_symbols.len() {
                let offset:(usize,usize) = borders[i].0;
                for pixel in pixels_in_symbols[i].iter() {
                    let x = pixel.0;
                    let y = pixel.1;
                    let out_pixel = symbol_images[i].get_pixel_mut((x-offset.0) as u32,(y-offset.1) as u32);
                    *out_pixel = image::Luma([255]);
                }
            }
            if debug_out { println!("{} : Created symbol images",time(start)); }
            return symbol_images;
        }
        
    }
    fn to_bin2dvec(debug_out:bool,width:usize,height:usize,img_raw:&mut Vec<u8>) -> Vec<Vec<u32>> {
        // 2d vector of size of image, where each pixel will be labelled as to which symbol it belongs
        let start = Instant::now();
        let mut symbols:Vec<Vec<u32>> = vec!(vec!(1u32;width as usize);height as usize);
        println!("width * height = length : {} * {} = {}|{}k|{}m",width,height,img_raw.len(),img_raw.len()/1000,img_raw.len()/1000000);
        
        let avg_luma:u8 = (img_raw.iter().fold(0u32,|sum,x| sum+*x as u32) / img_raw.len() as u32) as u8;
        println!("avg_luma:{}",avg_luma);
        
        for y in 0..height {
            for x in 0..width {
                let luma = img_raw[y*width+x];
                img_raw[y*width+x] = if luma < avg_luma-LUMA_ADJUSTMENT { 0 } else { 255 };
                symbols[y][x] = (img_raw[y*width+x] / 255) as u32;
            }
        }
        if debug_out { println!("{} : Converted image to binary",time(start)); }
        return symbols;
    }
    
    // Nicely prints Vec<Vec<u8>> as matrix
    #[allow(dead_code,non_snake_case)]
    pub fn symbols_intial_prt(matrix:&Vec<Vec<u32>>) -> () {

        println!();
        let shape = (matrix.len(),matrix[0].len()); // shape[0],shape[1]=row,column
        let spacing = 1*shape.0;
        horizontal_number_line(shape.0);

        println!("    ┌{:─<1$}┐","",spacing);
        for row in 0..shape.1 {
            vertical_number_line(row);

            print!("│");
            for col in 0..shape.0 {
                if matrix[col][row] == 1 { print!("{}",matrix[col][row]);/*print!("{}",WHITE_SPACE_SYMBOL);*/ }
                else { print!("{}",matrix[col][row]); }
                
                
            }
            println!("│");
        }
        println!("    └{:─<1$}┘","",spacing);
        print!("   {:<1$}","",(spacing/2)-1);
        println!("   [{},{}]",shape.0,shape.1);
        println!();

        fn horizontal_number_line(rows:usize) -> () {
            print!("\n   ");
            for col in 0..rows/10 {
                print!("{: <1$}","",4);
                print!("{: >2}",col);
                print!("{: <1$}","",4);
            }
            print!("\n    ");
            for _ in 0..rows/10 {
                print!("┌{:─<1$}┐","",8);
            }
            print!("┌{:─<1$}","",rows%10);
            print!("\n    ");
            for col in 0..rows {
                print!("{: >1}",col%10)
            }
            println!();
        }

        fn vertical_number_line(row:usize) -> () {
            if row % 10 == 5 {
                print!("{: >2}",row/ 10);
            } else { print!("  "); }

            if row % 10 == 0 {
                print!("┌");
            }
            else if row % 10 == 9 {
                print!("└");
            }
            else {
                print!("│");
            }
            print!("{}",row % 10);
        }
    }
    #[allow(dead_code,non_snake_case)]
    pub fn symbols_classified_prt(matrix:&Vec<Vec<u32>>) -> () {

        println!();
        let shape = (matrix.len(),matrix[0].len()); // shape[0],shape[1]=row,column
        let spacing = 2*shape.0;
        
        horizontal_number_line(shape.0);

        println!("    ┌─{:─<1$}┐","",spacing);
        for row in 0..shape.1 {
            vertical_number_line(row);

            print!("│");
            for col in 0..shape.0 {
                // TODO Do the whitespace print better
                if matrix[col][row] == 1 { print!(" {}",WHITE_SPACE_SYMBOL); }
                else { print!("{: >2}",matrix[col][row]); }
            }
            println!(" │");
        }
        println!("    └─{:─<1$}┘","",spacing);
        print!("{:<1$}","",(spacing/2)-1);
        println!("[{},{}]",shape.0,shape.1);
        println!();

        fn horizontal_number_line(rows:usize) -> () {
            print!("\n   ");
            for col in 0..rows/10 {
                print!("{: <1$}","",9);
                print!("{: >2}",col);
                print!("{: <1$}","",9);
            }
            print!("\n    ");
            for _ in 0..rows/10 {
                print!("┌{:─<1$}┐","",2*9);
            }
            print!("┌{:─<1$}","",rows%10);
            print!("\n    ");
            for col in 0..rows {
                print!("{: >2}",col%10)
            }
            println!();
        }

        fn vertical_number_line(row:usize) -> () {
            if row % 10 == 5 {
                print!("{: >2}",row/ 10);
            } else { print!("  "); }

            if row % 10 == 0 {
                print!("┌");
            }
            else if row % 10 == 9 {
                print!("└");
            }
            else {
                print!("│");
            }
            print!("{}",row % 10);
        }
    }
}
fn construct(debug_out:bool,classes:&[char],bounds:&Vec<((usize,usize),(usize,usize))>) -> String {
    #[derive(Copy,Clone,Default,Debug)]
    struct Point {
        x:usize,
        y:usize,
    }
    #[derive(Clone,Debug)]
    struct Symbol {
        class:String,
        bounds:(Point,Point)
    }
    #[derive(Debug)]
    struct Row {
        centre:usize,
        height:usize,
        sum:usize,
        symbols:Vec<Symbol>,
        scripts:(Option<*mut Row>,Option<*mut Row>),
        parent:Option<*mut Row>,
    }
    
    let start = Instant::now();

    let mut combined:Vec<Symbol> = 
        izip!(classes.iter(),bounds.iter())
        .map(|(class,bounds)| 
            Symbol {
                class: (*class).to_string(),
                bounds: (Point{x:(bounds.0).0,y:(bounds.0).1},Point{x:(bounds.1).0,y:(bounds.1).1})
            }
        ).collect();

    // Sort by min x, ordering symbols horizontally
    combined.sort_by(|a,b| ((a.bounds).0).x.cmp(&((b.bounds).0).x));

    // min_x after sorting, thus O(1) instead of O(n)
    let min_x:usize = ((combined[0].bounds).0).x;
    let min_y:usize = bounds.iter().fold(usize::MAX, |min,x| (if (x.0).1 < min { (x.0).1 } else { min }));
    // Subtract mins from all bounds
    for symbol in combined.iter_mut() {
        symbol.bounds.0.x -= min_x;
        symbol.bounds.0.y -= min_y;
        symbol.bounds.1.x -= min_x;
        symbol.bounds.1.y -= min_y;
    }

    let y_centers:Vec<usize> = combined.iter().map(|s| ((s.bounds.0).y+(s.bounds.1).y)/2).collect();
    let mut rows:Vec<Row> = vec![Row{
        centre:y_centers[0],
        height:usize::default(),
        sum:y_centers[0],
        symbols:vec![combined[0].clone()],
        scripts:(None,None),
        parent:None
    }];
    
    for i in 1..y_centers.len() {
        //println!("symbol: {}",combined[i].class);
        let mut new_row = true;
        for t in 0..rows.len() {
            // TODO Equations here could be done more efficiently, fix that.
            //println!("(1f32-({} as f32 / {} as f32)).abs() <= {} : {}",y_centers[i],rows[t].centre,ROW_CLEARANCE,(1f32-(y_centers[i] as f32 / rows[t].centre as f32)).abs());
            //print_row(&rows[t].symbols);
            if (1f32-(y_centers[i] as f32 / rows[t].centre as f32)).abs() <= ROW_CLEARANCE {
                rows[t].symbols.push(combined[i].clone());
                rows[t].sum += y_centers[i];
                rows[t].centre = rows[t].sum / rows[t].symbols.len();
                new_row = false;
                break;
            }
        }
        if new_row {
            rows.push(Row{
                centre:y_centers[i],
                height:usize::default(),
                sum:y_centers[i],
                symbols:vec![combined[i].clone()],
                scripts:(None,None),
                parent: None,
            });
            // print!("row centres: [ ");
            // for row in rows.iter() {
            //     print!("{} ",row.centre);
            // }
            // println!("]");
        }
    }

    if debug_out {
        println!("rows:");
        for row in rows.iter() {
            print_row(&row.symbols);
            println!();
        }
    }
    

    // Construct composite symbols
    for row in rows.iter_mut() {
        let mut i = 0usize;
        while i < row.symbols.len() {
            if row.symbols[i].class == "-" {
                if row.symbols[i+1].class == "-" {
                    // println!("{}/{}={}",row.symbols[i].bounds.0.x,row.symbols[i+1].bounds.0.x,row.symbols[i].bounds.0.x as f32 / row.symbols[i+1].bounds.0.x as f32);
                    if (1f32 - row.symbols[i].bounds.0.x as f32 / row.symbols[i+1].bounds.0.x as f32).abs() <= 0.2f32 {
                        row.symbols[i].class="=".to_string();
                        // print_row(&row.symbols);
                        row.symbols[i].bounds = (
                            Point{x:min(row.symbols[i].bounds.0.x,row.symbols[i+1].bounds.0.x),y:min(row.symbols[i].bounds.0.y,row.symbols[i+1].bounds.0.y)},
                            Point{x:max(row.symbols[i].bounds.1.x,row.symbols[i+1].bounds.1.x),y:min(row.symbols[i].bounds.1.y,row.symbols[i+1].bounds.1.y)}
                        );
                        row.symbols.remove(i+1);
                    }
                }
                else if row.symbols[i+1].class == "." && row.symbols[i+2].class == "." {
                    if within_x(&row.symbols[i+1],&row.symbols[i]) && within_x(&row.symbols[i+2],&row.symbols[i]) {
                        row.symbols[i].class = "\\div".to_string(); // /div
                        row.symbols.remove(i+1);
                        row.symbols.remove(i+1); // After first remove now i+1 == prev i+2
                    }
                }
            }
            i += 1;
        }
    }

    if debug_out {
        println!("symbol rows:");
        for row in rows.iter() {
            print_row(&row.symbols);
            println!();
        }
    }
    

    // Sorts rows in vertical order rows[0] is bottom row
    rows.sort_by(|a,b| (b.centre.cmp(&a.centre)));

    if debug_out {
        println!("vertically ordered rows:");
        for row in rows.iter() {
            print_row(&row.symbols);
            println!();
        }
    }

    

    // Average height of rows
    for row in rows.iter_mut() {
        let mut ignored_symbols = 0usize;
        for symbol in row.symbols.iter() {
            // Ignore the heights of '-' and '.' since they will throw off the average
            match symbol.class.as_str() {
                "-" | "." => ignored_symbols +=1,
                _ => row.height += symbol.bounds.1.y - symbol.bounds.0.y,
            }
        }
        // Average height in row
        row.height /= row.symbols.len()-ignored_symbols;
    }

    if debug_out {
        print!("row heights:");
        for row in rows.iter() {
            print!("{}",row.height);
        }
        println!();
    }
    

    // Average height of all rows
    //let avg_height = row_heights.iter().fold(0usize,|sum,x| sum+x) / row_heights.len();

    //println!("got here");

    let mut row_scripts:Vec<(Option<usize>,Option<usize>)> = vec!((None,None);rows.len()); // .0=subscript, .1=superscript

    let mut unassigned_rows:Vec<&mut Row> = rows.iter_mut().map(|x|x).collect();

    while unassigned_rows.len() > 1 {
        // println!("------------------------------------------------");
        // println!("unassigned_rows.len(): {}",unassigned_rows.len());
        // println!("unassigned_rows:");
        // for unassigned_row in unassigned_rows.iter() {
        //     println!("{:.?}",unassigned_row);
        // }
        // List of indexes in reference to rows to remove from unassigned_rows as they have been assigned
        let mut removal_list:Vec<usize> = Vec::new();
        for i in 0..unassigned_rows.len() {
            //println!("------------------------");
            //println!("\ti:{}, symbols: {:.?}",i,unassigned_rows[i].symbols);
            // If there are rows above
            let mut pos_sup = false; //Possible superscript
            if i > 0 {
                //println!("i-1={}",i-1);
                // If height of this row less than the row below
                if unassigned_rows[i].height < unassigned_rows[i-1].height {
                    pos_sup = true;
                }
            }
            let mut pos_sub = false; //Possible subscript
            if i < unassigned_rows.len()-1 {
                //println!("i+1={}",i+1);
                // If height of this row less than the row above
                if unassigned_rows[i].height < unassigned_rows[i+1].height {
                    pos_sub = true;
                }
            }
            // TODO This is ugly, is there a better way?
            let pointer:*mut Row = *unassigned_rows.get_mut(i).unwrap() as *mut Row;
            // If could both be superscript and subscript
            if pos_sup && pos_sub {
                // Belongs to row with shortest height as superscript or subscript
                if unassigned_rows[i+1].height < unassigned_rows[i-1].height {
                    unassigned_rows[i+1].scripts.0 = Some(pointer); // Sets subscript link
                    unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i+1).unwrap() as *mut Row); // Sets link to parent
                    //println!("\toutcome 1");
                    removal_list.push(i);
                }
                // This `else if` could simply be `else` but it doesn't need to be performant and is easier to understand this way.
                else if unassigned_rows[i-1].height < unassigned_rows[i+1].height {
                    unassigned_rows[i-1].scripts.1 = Some(pointer); // Sets superscript link
                    unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i-1).unwrap() as *mut Row); // Sets link to parent
                    //println!("\toutcome 2");
                    removal_list.push(i);
                }
            }
            else if pos_sup {
                unassigned_rows[i-1].scripts.1 = Some(pointer); // Sets superscript link
                unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i-1).unwrap() as *mut Row); // Sets link to parent
                //println!("\toutcome 3");
                removal_list.push(i);
            }
            // This `else if` could simply be `else` but it doesn't need to be performant and is easier to understand this way.
            else if pos_sub {
                unassigned_rows[i+1].scripts.0 = Some(pointer); // Sets subscript link
                unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i+1).unwrap() as *mut Row); // Sets link to parent
                //println!("\toutcome 4");
                removal_list.push(i);
            }
        }
        //println!("removal_list: {:.?}",removal_list);
        // Removes assigned rows from `unassigned_rows`
        remove_indexes(&mut unassigned_rows, &removal_list);
        //println!("unassigned_rows.len(): {}",unassigned_rows.len());
    }
    //println!("finished script setting");

    // unsafe {
    //     println!("\nrows:");
    //     for row in rows.iter() {
    //         println!();
    //         println!("{:.?}",row);
    //         if let Some(pointer) = row.scripts.0 {
    //             println!("{:.?} -> {:.?}",pointer,(*pointer).symbols)
    //         }
    //         if let Some(pointer) = row.scripts.1 {
    //             println!("{:.?} -> {:.?}",pointer,(*pointer).symbols)
    //         }
    //     }
    // }
    
    let mut current_row:&mut Row = unassigned_rows.get_mut(0).unwrap();
    let mut latex:String = String::from(format!("{}",current_row.symbols[0].class));
    current_row.symbols.remove(0);
    unsafe {
        loop  {
            if debug_out { println!("building: {}",latex); }
            // TODO Make `min_sub` and `min_sup` immutable
            // Gets min x bound of symbol in subscript row
            let mut min_sub:usize = usize::max_value();
            if let Some(sub_row) = current_row.scripts.0 {
                if let Some(symbol) = (*sub_row).symbols.first() {
                    min_sub = symbol.bounds.0.x;
                }
            }
            // Gets min x bound of symbol in superscript row
            let mut min_sup:usize = usize::max_value();
            if let Some(sub_row) = current_row.scripts.1 {
                if let Some(symbol) = (*sub_row).symbols.first() {
                    min_sup = symbol.bounds.0.x;
                }
            }
            // Gets min x bound of next symbol in current row
            let min_cur:usize = if let Some(symbol) = current_row.symbols.get(0) {
                symbol.bounds.0.x
            } else { usize::max_value() };

            let mut min_par:usize = usize::max_value();
            if let Some(parent) = current_row.parent {
                if let Some(symbol) = (*parent).symbols.first() {
                    min_par = symbol.bounds.0.x;
                }
            }
    
            //println!("(sub,sup,cur,par):({},{},{},{})",min_sub,min_sup,min_cur,min_par);

            if let Some(min) = min_option(&[min_sub,min_sup,min_cur,min_par]) {
                if min == min_par {
                    current_row = &mut *current_row.parent.unwrap();
                    latex.push_str("}");
                } else {
                    if min == min_sub {
                        current_row = &mut *current_row.scripts.0.unwrap();
                        latex.push_str(&format!("_{{{}",current_row.symbols[0].class));
                    }
                    else if min == min_sup {
                        current_row = &mut *current_row.scripts.1.unwrap();
                        latex.push_str(&format!("^{{{}",current_row.symbols[0].class));     
                    }
                    else if min == min_cur {
                        latex.push_str(&current_row.symbols[0].class);
                    }
                    current_row.symbols.remove(0);
                }
                
                
            } else {
                if let Some(parent) = current_row.parent {
                    current_row = &mut *parent;
                    latex.push_str("}");
                }
                else {
                    break;
                }
                
            }
        }
    }

    if debug_out { println!("{} : Construction finished",time(start)); }

    return latex;

    // TODO Should I use a template when I don't need it?
    fn min_option(slice:&[usize]) -> Option<usize> {
        let mut min = usize::max_value();
        for val in slice {
            if *val < min { 
                min = *val;
            }
        }
        if min == usize::max_value() { return None; }
        return Some(min);
    }

    fn print_row(symbols:&Vec<Symbol>) {
        print!("[ ");
        for symbol in symbols {
            print!("{} ",symbol.class);
        }
        print!("]");
    }
    // If inner is within x bounds of outer
    fn within_x(inner:&Symbol,outer:&Symbol) -> bool {
        if inner.bounds.0.x > outer.bounds.0.x && inner.bounds.1.x < outer.bounds.1.x {
            return true;
        }
        return false;
    }
    fn min(a:usize,b:usize) -> usize {
        if a < b { return a; }
        return b;
    }
    fn max(a:usize,b:usize) -> usize {
        if a > b { return a; }
        return b;
    }

    /*
    let mut grid = vec!(vec!(None;combined.len());3);

    for (i,(class,bounds)) in combined.iter().enumerate() {
        // y upper bound > average y -> superscript
        //println!("char:{},min_y:{},max_y:{}",class,(bounds.0).1,(bounds.1).1);
        if 
        if (bounds.0).1 > avg_y { 
            grid[2][i] = Some(class);
        }
        else if (bounds.1).1 < avg_y {
            grid[0][i] = Some(class);
        }
        else {
            grid[1][i] = Some(class);
        }
    }

    if debug_out { println!("grid:\n{:.?}",grid); }

    let mut latex:String = String::new();
    for i in 0..combined.len() {
        if let Some(superscript) = grid[0][i] {
            println!("^{}",superscript);
            latex.push_str(&format!("^{}",superscript));
        }
        else if let Some(subscript) = grid[2][i] {
            println!("_{}",subscript);
            latex.push_str(&format!("_{}",subscript));
        }
        else if let Some(symbol) = grid[1][i] { // Option here should always be `Some(symbol)`
            println!("{}",*symbol);
            latex.push(*symbol);
        }
    }

    return latex;
    */

    fn remove_indexes<T>(vec:&mut Vec<T>,indxs:&[usize]) {
        let mut counter_actor = 0usize;
        for indx in indxs {
            vec.remove(indx-counter_actor);
            counter_actor += 1;
        }
    }
}
fn time(instant:Instant) -> String {
    let mut millis = instant.elapsed().as_millis();
    let seconds = (millis as f32 / 1000f32).floor();
    millis = millis % 1000;
    let time = format!("{:#02}:{:#03}",seconds,millis);
    return time;
}