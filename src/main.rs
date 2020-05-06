use std::time::Instant;

use itertools::izip;
use std::io::Read;
use cogent::core::*;
use image::{ImageBuffer,Luma};
use std::path::Path;
use std::fs;
use std::fs::File;
use std::collections::VecDeque;
use image::imageops::FilterType;
use std::usize;
use std::cmp;
use std::collections::HashMap;
use std::ops::{Sub,SubAssign,Add};

use arrayfire::{Array,max,eq,sum,transpose,Dim4,print_gen,af_print};

const SCALE:u32 = 20u32;
const LUMA_ADJUSTMENT:u8 = 15u8; // Luma less than set to 0 and more than set to 255.
const ROW_CLEARANCE:f32 = 0.3f32;
const COVERED_PIXEL:u8 = 2u8; // Can be any u8 value, as long as not 

fn main() {
    // Runs segmentation
    // -----------------
    let segment_1 = segment(true,"1.jpg");
    let segment_2 = segment(false,"2.jpg");
    let segment_3 = segment(false,"3.jpg");
    let segment_4 = segment(false,"4.jpg");
    let segment_5 = segment(false,"5.jpg");
    let segment_6 = segment(false,"6.jpg");
    let segment_7 = segment(false,"7.jpg");
    let segment_8 = segment(false,"8.jpg");
    let segment_9 = segment(false,"9.jpg");
    let segment_10 = segment(false,"10.jpg");
    let segment_11 = segment(false,"11.jpg");
    let segment_12 = segment(false,"12.jpg");
    let segment_13 = segment(false,"13.jpg");
    let segment_14 = segment(false,"14.jpg");
    let segment_15 = segment(false,"15.jpg");

    let segmentat_alphabet = segment(false,"alphabet.jpg");

    // Manaully set correct classes for construction testing
    // -----------------
    let correct_classes_1:[&str;15] = ["3","2","x","7","1","2","\\cdot ","b","+","-","y","-","-","-","\\cdot "];
    let correct_classes_2:[&str;10] = ["\\sqrt{} ","2","b","2","a","-","c","-","\\cdot ","-"];
    let correct_classes_3:[&str;11] = ["2","2","c","b","-","2","+","-","a","\\cdot ","\\cdot "];
    let correct_classes_4:[&str;9] = ["4","\\cdot ","2","\\codt ","2","\\cdot ","-","\\cdot ","-"];
    let correct_classes_5:[&str;7] = ["\\sqrt{} ","\\cdot ","2","4","-","-","\\cdot "];
    let correct_classes_6:[&str;6] = ["7","<","5","9",">","\\cdot "];
    let correct_classes_7:[&str;8] = ["6","8","5","9","+","\\cdot ","7","8"];
    let correct_classes_8:[&str;15] = ["4","1","2","7","3","2","5","+","3","1","-","4","3","\\cdot","\\cdot"];
    let correct_classes_9:[&str;16] = ["5","4","3","\\cdot ","\\times ","\\cdot ","2","-","1","Y","+","-","\\cdot ","\\cdot ","\\cdot ","\\cdot "];
    let correct_classes_10:[&str;8] = ["2","x","\\cdot ","-","y","\\cdot ","\\cdot ","-"];
    let correct_classes_11:[&str;9] = ["1","2","-","3","4","2","2","x","-"];
    let correct_classes_12:[&str;11] = ["3","5","-","4","x","+","2","2","y","x","-"];
    let correct_classes_13:[&str;12] = ["2","2","b","b","+","-","+","y","x","2","1","\\cdot "];
    let correct_classes_14:[&str;7] = ["1","2","2","X","2","-","1"];
    let correct_classes_15:[&str;7] = ["1","2","2","x","2","-","1"];

    // Gets bounds
    // -----------------
    let bounds_1:Vec<((usize,usize),(usize,usize))> = segment_1.iter().map(|&(_,b)| b).collect();
    let bounds_2:Vec<((usize,usize),(usize,usize))> = segment_2.iter().map(|&(_,b)| b).collect();
    let bounds_3:Vec<((usize,usize),(usize,usize))> = segment_3.iter().map(|&(_,b)| b).collect();
    let bounds_4:Vec<((usize,usize),(usize,usize))> = segment_4.iter().map(|&(_,b)| b).collect();
    let bounds_5:Vec<((usize,usize),(usize,usize))> = segment_5.iter().map(|&(_,b)| b).collect();
    let bounds_6:Vec<((usize,usize),(usize,usize))> = segment_6.iter().map(|&(_,b)| b).collect();
    let bounds_7:Vec<((usize,usize),(usize,usize))> = segment_7.iter().map(|&(_,b)| b).collect();
    let bounds_8:Vec<((usize,usize),(usize,usize))> = segment_8.iter().map(|&(_,b)| b).collect();
    let bounds_9:Vec<((usize,usize),(usize,usize))> = segment_9.iter().map(|&(_,b)| b).collect();
    let bounds_10:Vec<((usize,usize),(usize,usize))> = segment_10.iter().map(|&(_,b)| b).collect();
    let bounds_11:Vec<((usize,usize),(usize,usize))> = segment_11.iter().map(|&(_,b)| b).collect();
    let bounds_12:Vec<((usize,usize),(usize,usize))> = segment_12.iter().map(|&(_,b)| b).collect();
    let bounds_13:Vec<((usize,usize),(usize,usize))> = segment_13.iter().map(|&(_,b)| b).collect();
    let bounds_14:Vec<((usize,usize),(usize,usize))> = segment_14.iter().map(|&(_,b)| b).collect();
    let bounds_15:Vec<((usize,usize),(usize,usize))> = segment_15.iter().map(|&(_,b)| b).collect();

    // Gets LaTeX usualy correctly set classes
    // -----------------
    let correct_latex_1 = construct(true,&correct_classes_1,&bounds_1);
    let correct_latex_2 = construct(false,&correct_classes_2,&bounds_2);
    let correct_latex_3 = construct(false,&correct_classes_3,&bounds_3);
    let correct_latex_4 = construct(false,&correct_classes_4,&bounds_4);
    let correct_latex_5 = construct(false,&correct_classes_5,&bounds_5);
    let correct_latex_6 = construct(false,&correct_classes_6,&bounds_6);
    let correct_latex_7 = construct(false,&correct_classes_7,&bounds_7);
    let correct_latex_8 = construct(false,&correct_classes_8,&bounds_8);
    let correct_latex_9 = construct(false,&correct_classes_9,&bounds_9);
    let correct_latex_10 = construct(false,&correct_classes_10,&bounds_10);
    let correct_latex_11 = construct(false,&correct_classes_11,&bounds_11);
    let correct_latex_12 = construct(false,&correct_classes_12,&bounds_12);
    let correct_latex_13 = construct(false,&correct_classes_13,&bounds_13);
    let correct_latex_14 = construct(false,&correct_classes_14,&bounds_14);
    let correct_latex_15 = construct(false,&correct_classes_15,&bounds_15);

    // Prints LaTeX
    // -----------------
    println!("correct_latex_1 :{}",correct_latex_1);
    println!("correct_latex_2 :{}",correct_latex_2);
    println!("correct_latex_3 :{}",correct_latex_3);
    println!("correct_latex_4 :{}",correct_latex_4);
    println!("correct_latex_5 :{}",correct_latex_5);
    println!("correct_latex_6 :{}",correct_latex_6);
    println!("correct_latex_7 :{}",correct_latex_7);
    println!("correct_latex_8 :{}",correct_latex_8);
    println!("correct_latex_9 :{}",correct_latex_9);
    println!("correct_latex_10:{}",correct_latex_10);
    println!("correct_latex_11:{}",correct_latex_11);
    println!("correct_latex_12:{}",correct_latex_12);
    println!("correct_latex_13:{}",correct_latex_13);
    println!("correct_latex_14:{}",correct_latex_14);
    println!("correct_latex_15:{}",correct_latex_15);

    // Gets pixels for symbols for images
    // -----------------
    let input_1:Vec<Vec<f32>> = segment_1.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_2:Vec<Vec<f32>> = segment_2.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_3:Vec<Vec<f32>> = segment_3.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_4:Vec<Vec<f32>> = segment_4.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_5:Vec<Vec<f32>> = segment_5.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_6:Vec<Vec<f32>> = segment_6.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_7:Vec<Vec<f32>> = segment_7.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_8:Vec<Vec<f32>> = segment_8.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_9:Vec<Vec<f32>> = segment_9.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_10:Vec<Vec<f32>> = segment_10.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_11:Vec<Vec<f32>> = segment_11.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_12:Vec<Vec<f32>> = segment_12.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_13:Vec<Vec<f32>> = segment_13.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_14:Vec<Vec<f32>> = segment_14.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();
    let input_15:Vec<Vec<f32>> = segment_15.iter().map(|(pixels,_)| pixels.iter().map(|&x|x as f32).collect()).collect();

    // Loads neural network
    //  1st bool determines if it is trained or simply loaded from file
    //  2nd bool determines if to print an evaluation of class accuracies
    let net = train_net(false,false);

    // Runs symbols through network
    // -----------------
    let class_labels_1:Vec<usize> = net.run(&input_1);
    let class_labels_2:Vec<usize> = net.run(&input_2);
    let class_labels_3:Vec<usize> = net.run(&input_3);
    let class_labels_4:Vec<usize> = net.run(&input_4);
    let class_labels_5:Vec<usize> = net.run(&input_5);
    let class_labels_6:Vec<usize> = net.run(&input_6);
    let class_labels_7:Vec<usize> = net.run(&input_7);
    let class_labels_8:Vec<usize> = net.run(&input_8);
    let class_labels_9:Vec<usize> = net.run(&input_9);
    let class_labels_10:Vec<usize> = net.run(&input_10);
    let class_labels_11:Vec<usize> = net.run(&input_11);
    let class_labels_12:Vec<usize> = net.run(&input_12);
    let class_labels_13:Vec<usize> = net.run(&input_13);
    let class_labels_14:Vec<usize> = net.run(&input_14);
    let class_labels_15:Vec<usize> = net.run(&input_15);

    // Lodas hash map which maps class index labels to symbols
    let file = File::open("label_mappings.json");
    let mut string_contents:String = String::new();
    file.unwrap().read_to_string(&mut string_contents).unwrap();
    let label_mappings:HashMap<usize,&str> = serde_json::from_str(&string_contents).unwrap();

    //println!("got here");

    // Gets class symbols
    // -----------------
    let class_symbols_1:Vec<&str> = class_labels_1.into_iter().map(|x| *label_mappings.get(&x).unwrap()).collect();
    let class_symbols_2:Vec<&str> = class_labels_2.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_3:Vec<&str> = class_labels_3.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_4:Vec<&str> = class_labels_4.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_5:Vec<&str> = class_labels_5.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_6:Vec<&str> = class_labels_6.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_7:Vec<&str> = class_labels_7.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_8:Vec<&str> = class_labels_8.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_9:Vec<&str> = class_labels_9.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_10:Vec<&str> = class_labels_10.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_11:Vec<&str> = class_labels_11.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_12:Vec<&str> = class_labels_12.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_13:Vec<&str> = class_labels_13.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_14:Vec<&str> = class_labels_14.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();
    let class_symbols_15:Vec<&str> = class_labels_15.into_iter().map(|x|*label_mappings.get(&x).unwrap()).collect();

    // Prints class symbols
    // -----------------
    println!("class_symbols_1 :{:.?}",class_symbols_1);
    println!("class_symbols_2 :{:.?}",class_symbols_2);
    println!("class_symbols_3 :{:.?}",class_symbols_3);
    println!("class_symbols_4 :{:.?}",class_symbols_4);
    println!("class_symbols_5 :{:.?}",class_symbols_5);
    println!("class_symbols_6 :{:.?}",class_symbols_6);
    println!("class_symbols_7 :{:.?}",class_symbols_7);
    println!("class_symbols_8 :{:.?}",class_symbols_8);
    println!("class_symbols_9 :{:.?}",class_symbols_9);
    println!("class_symbols_10:{:.?}",class_symbols_10);
    println!("class_symbols_11:{:.?}",class_symbols_11);
    println!("class_symbols_12:{:.?}",class_symbols_12);
    println!("class_symbols_13:{:.?}",class_symbols_13);
    println!("class_symbols_14:{:.?}",class_symbols_14);
    println!("class_symbols_15:{:.?}",class_symbols_15);
}

fn train_net(train:bool,eval:bool) -> NeuralNetwork {
    //Setup
    let mut data = get_combined("combined_dataset");

    if train {
        let k = 55usize;
        let mut net = NeuralNetwork::new(data[0].0.len(),&[
            Layer::new(500,Activation::ReLU),
            Layer::new(100,Activation::ReLU),
            Layer::new(k,Activation::Softmax)
        ]);
        //Execution
        net.train(&data)
            .learning_rate(0.05)
            .tracking()
            .log_interval(MeasuredCondition::Iteration(1))
            .checkpoint_interval(MeasuredCondition::Iteration(1))
            .evaluation_data(EvaluationData::Percent(0.05))
            .halt_condition(HaltCondition::Iteration(30)).early_stopping_condition(MeasuredCondition::Iteration(100))
            .l2(0.1)
        .go();
        net.export("final classifier");
    }
    
    let net = NeuralNetwork::import("final classifier");

    if eval {
        data.sort_by(|(_,a),(_,b)| a.cmp(b));
        
        let (mut lower_indx,mut upper_indx) = (0usize,1usize);
        let mut class_slices:Vec<&[(Vec<f32>,usize)]> = Vec::new();
        
        loop {
            //println!("data[upper_indx-1].1:{}",data[upper_indx-1].1);
            while data[upper_indx].1 == data[upper_indx-1].1 {
                upper_indx += 1;
                if upper_indx == data.len() { break; }
            }
            println!("{} : size:{}",class_slices.len(),upper_indx-lower_indx);
            class_slices.push(&data[lower_indx..upper_indx]);
            
            if upper_indx == data.len() { break; }
            lower_indx = upper_indx;
            upper_indx += 1;
        }

        for class_slice in class_slices {
            let (input,_) = matrixify_inputs(class_slice);
            let outputs = net.inner_run(&input);

            let maxs:Array<f32> = max(&outputs,1i32);
            let class_vectors:Array<bool> = eq(&outputs,&maxs,true);
            

            let matrix = sum(&class_vectors,0i32).cast::<f32>();
            //af_print!("matrix:",matrix);
            let percentage_matrix =  matrix / (class_slice.len() as f32);
            //af_print!("percentage_matrix:",percentage_matrix);

            let mut vector_container = vec!(f32::default();percentage_matrix.dims().get()[1] as usize);
            percentage_matrix.host(&mut vector_container);
            //println!("vector_container: {:.?}",vector_container);
            //print!("dims:{}",percentage_matrix.dims());
            let class = class_slice[0].1;
            print!("{:02} : {:.2} | ",class,vector_container[class]);
            for i in 0..vector_container.len() {
                if i == class { continue; }
                if vector_container[i] > 0.05f32 { print!(" ({:02}:{:.2})",i,vector_container[i]) }
            }
            //println!("\nmat row: {:.2?}",vector_container);
            println!();
            //return;
        }
    }

    return net;

    fn get_combined<T:From<u8>>(path:&str) -> Vec<(Vec<T>,usize)> {
        let mut file = fs::File::open(path).unwrap();
        let mut combined_buffer = Vec::new();
        file.read_to_end(&mut combined_buffer).expect("Couldn't read combined");
    
        let mut combined_vec:Vec<(Vec<T>,usize)> = Vec::new();
        let multiplier:usize = (20*20)+1;
        let length = combined_buffer.len() / multiplier;

        for i in (0..length).rev() {
            let image_index = i * multiplier;
            let label_index = image_index + multiplier - 1;
    
            let label:u8 = combined_buffer.split_off(label_index)[0]; // Array is only 1 element regardless
    
            let image:Vec<u8> = combined_buffer.split_off(image_index);
            let image_f32 = image.iter().map(|&x| T::from(x)).collect();
    
            combined_vec.push((image_f32,label as usize));
        }
        return combined_vec;
    }

    fn matrixify_inputs(examples:&[(Vec<f32>,usize)]) -> (Array<f32>,Array<u32>) { // Array(in,examples,1,1), Array(examples,1,1,1)
        let in_len = examples[0].0.len();
        let example_len = examples.len();

        // Flattens examples into `in_vec` and `out_vec`
        let in_vec:Vec<f32> = examples.iter().flat_map(|(input,_)| input.clone() ).collect();
        let out_vec:Vec<u32> = examples.iter().map(|(_,class)| class.clone() as u32 ).collect();

        let input:Array<f32> = transpose(&Array::<f32>::new(&in_vec,Dim4::new(&[in_len as u64,example_len as u64,1,1])),false);
        let output:Array<u32> = Array::<u32>::new(&out_vec,Dim4::new(&[example_len as u64,1,1,1]));

        return (input,output);
    }
}
// Returns symbol images and bounds
fn segment(debug_out:bool,path:&str) -> Vec<(Vec<u8>,((usize,usize),(usize,usize)))> {
    #[derive(Clone,Eq,PartialEq)]
    enum Pixel { White,Black,Assigned }

    let start = Instant::now();

    // Open the image to segment (in this case it will reside within the `test_imgs` directory)
    let img = image::open(format!("test_imgs/{}",path)).unwrap().to_luma();
    
    // Gets dimensions of the image
    let dims = img.dimensions();
    let (width,height) = (dims.0 as usize,dims.1 as usize);
    
    // Gets raw pixel values from the image.
    let mut img_raw:Vec<u8> = img.clone().into_raw();

    // 2d vector of size of image, where each pixel will be labelled white/black (and later used in forest fire)
    let mut pixels:Vec<Vec<Pixel>> = raw_to_binary_2d_vector(debug_out,width,height,&mut img_raw);

    // Gets name of image file ('some_image.jpg' -> 'some_image')
    let name = path.split(".").collect::<Vec<&str>>()[0];

    // Outputs binary image
    if debug_out {
        if !Path::new("binary_imgs").exists() {
            fs::create_dir("binary_imgs").unwrap();
        }
        let binary_image = ImageBuffer::<Luma<u8>,Vec<u8>>::from_raw(width as u32,height as u32,img_raw.clone()).unwrap();
        binary_image.save(format!("binary_imgs/{}.png",name)).unwrap();
    }

    // Gets lists of pixels belonging to each symbol
    let pixel_lists:Vec<Vec<(usize,usize)>> = get_pixels_in_symbols(debug_out,width,height,&mut pixels);

    // Gets bounds, square bounds and square bounds scaling property for each symbol
    let bounds:Vec<(((usize,usize),(usize,usize)),((i32,i32),(i32,i32),i32))> = get_bounds(debug_out,&pixel_lists);

    // Outputs borders image
    if debug_out { output_bounds(2,[255,0,0],path,name,&img_raw,&bounds,width,height); }

    // Gets scaled pixels beloning to each symbol
    let symbols:Vec<Vec<u8>> = get_symbols(debug_out,&pixel_lists,&bounds);

    // Outputs symbol images
    if debug_out { output_symbols(&symbols,name); }
    
    if debug_out { println!("{} : Finished segmentation",time(start)); }

    return izip!(symbols,bounds).map(|(s,b)|(s,b.0)).collect();

    fn raw_to_binary_2d_vector(debug_out:bool,width:usize,height:usize,img_raw:&mut Vec<u8>) -> Vec<Vec<Pixel>> {
        let start = Instant::now();

        // Intialises 2d vector of size height*width with Pixel::White.
        let mut pixels:Vec<Vec<Pixel>> = vec!(vec!(Pixel::White;width as usize);height as usize);

        if debug_out {
            println!(
                "width * height = length : {} * {} = {}|{}k|{}m",
                width,height,img_raw.len(),img_raw.len()/1000,img_raw.len()/1000000
            ); 
        }
        
        // Gets average luma among pixels
        let avg_luma:u8 = (img_raw.iter().fold(0u32,|sum,&x| sum + x as u32) / img_raw.len() as u32) as u8;
        //  Uses `.fold()` instead of `.sum()` since sum of values will easily exceed `u8:MAX`
    
        if let Some(boundary) =  avg_luma.checked_sub(LUMA_ADJUSTMENT) {
            for y in 0..height {
                for x in 0..width {
                    let indx = y*width+x; // Index of pixel of coordinates (x,y) in `img_raw`
                    let luma = img_raw[indx];
                    if luma < boundary {
                        img_raw[indx] = 0;
                        pixels[y][x] = Pixel::Black;
                    } else {
                        img_raw[indx] = 255;
                    }
                }
            }
        }
        else {
            panic!("Average luminance of image too low");
        }

        if debug_out { println!("{} : Converted image to binary",time(start)); }

        return pixels;
    }
    fn get_pixels_in_symbols(debug_out:bool,width:usize,height:usize,pixels:&mut Vec<Vec<Pixel>>) -> Vec<Vec<(usize,usize)>> {
        let start = Instant::now();

        // List of lists of pixels belonging to each symbol.
        let mut pixel_lists:Vec<Vec<(usize,usize)>> = Vec::new();

        // Iterates through pixels
        for y in 0..height {
            for x in 0..width {
                if pixels[y][x] == Pixel::Black {
                    // Pushes new list to hold pixels belonging to this newly found symbol
                    pixel_lists.push(Vec::new());
                    
                    // Triggers the forest fire algorithm
                    let last_index = pixel_lists.len()-1;
                    forest_fire(pixels,width,height,x,y,&mut pixel_lists[last_index]);
                }
            }
        }

        if debug_out { println!("{} : Fill finished",time(start)); }

        return pixel_lists;

        fn forest_fire(
            pixels:&mut Vec<Vec<Pixel>>,
            width:usize,height:usize,
            x:usize,y:usize,
            pixel_list:&mut Vec<(usize,usize)>
        ) {
            // Push 1st pixel to symbol
            pixel_list.push((x,y));
            // Sets 1st pixels symbol number
            pixels[y][x] = Pixel::Assigned;
            // Initialises queue for forest fire
            let mut queue:VecDeque<(usize,usize)> = VecDeque::new();
            // Pushes 1st pixel to queue
            queue.push_back((x,y));

            // While there is value in queue (effectively while !queue.is_empty()) 
            while let Some(n) = queue.pop_front() {
                // +x
                if n.0 < width-1 {
                    let (x,y) = (n.0+1,n.1);
                    pixels[y][x] = run_pixel(x,y,pixels,pixel_list,&mut queue);
                }
                // -x
                if n.0 > 0 {
                    let (x,y) = (n.0-1,n.1);
                    pixels[y][x] = run_pixel(x,y,pixels,pixel_list,&mut queue);
                }
                // +y
                if n.1 < height-1 {
                    let (x,y) = (n.0,n.1+1);
                    pixels[y][x] = run_pixel(x,y,pixels,pixel_list,&mut queue);
                }
                // -y
                if n.1 > 0 {
                    let (x,y) = (n.0,n.1-1);
                    pixels[y][x] = run_pixel(x,y,pixels,pixel_list,&mut queue);
                }
            }
            fn run_pixel(
                x:usize,
                y:usize,
                pixels:&Vec<Vec<Pixel>>,
                pixel_list:&mut Vec<(usize,usize)>,
                queue:&mut VecDeque<(usize,usize)>
            ) -> Pixel {
                if pixels[y][x] == Pixel::Black { // If black pixel unassigned to symbol
                    pixel_list.push((x,y)); // Push pixel to symbol
                    queue.push_back((x,y)); // Enqueue pixel for forest fire algorithm
                    return Pixel::Assigned; // Return value to set `symbols[y][x]` to
                }
                return Pixel::White; // Return value to set `symbols[y][x]` to
            }
        }
    }

    // Returns tupel of: 2d vec of pixels in symbol, Bounds of symbol
    fn get_symbols(
        debug_out:bool,
        pixel_lists:&Vec<Vec<(usize,usize)>>,
        bounds:&Vec<(((usize,usize),(usize,usize)),((i32,i32),(i32,i32),i32))>
    ) -> Vec<Vec<u8>> {
        let start = Instant::now();
        let mut symbols:Vec<Vec<u8>> = Vec::with_capacity(pixel_lists.len());
    
        for i in 0..pixel_lists.len() {
            // Sets `min_actuals` as minimum bounds and `bounds` as square bounds.
            let ((min_actuals,_),bounds) = bounds[i];
    
            // Calculates height and width of image using sqaure bounds
            let height:usize = ((bounds.1).1 - (bounds.0).1 + 1) as usize;
            let width:usize = ((bounds.1).0 - (bounds.0).0 + 1) as usize;
    
            // Constructs list to hold symbol image
            let mut symbol = vec!(vec!(255u8;width);height);
    
            // Iterates over pixels belonging to symbol
            for &(x,y) in pixel_lists[i].iter() {      
                // Sets x,y coordinates scaled to square bounds from original x,y coordinates.
                let (nx,ny) = if bounds.2 < 0 { ((x as i32-bounds.2) as usize,y) } else { (x,(y as i32 + bounds.2) as usize) };
    
                // Sets pixel in symbol image list 
                symbol[ny-min_actuals.1][nx-min_actuals.0] = 0u8;
            }
    
            // Constructs image buffer from symbol vector
            let mut symbol_image = ImageBuffer::<Luma<u8>,Vec<u8>>::from_raw(
                width as u32,
                height as u32,
                symbol.into_iter().flatten().collect()
            ).unwrap();
    
            // Scales image buffer
            let scaled_image = image::imageops::resize(&mut symbol_image,SCALE,SCALE,FilterType::Triangle);
    
            // Sets list to image buffer and carries out binarization.
            //  Running basic binarization here is necessary as scaling will have blurred some pixels.
            //  Basic binarization should also be proficient as the blurring will be minor.
            let binary_vec:Vec<u8> = scaled_image.into_raw().into_iter().map(|p| if p<220 { 0 } else { 1 }).collect();
    
            // Pushes the scaled symbol list to the symbols list.
            symbols.push(binary_vec);
        }
        if debug_out { println!("{} : Symbols set",time(start)); };
    
        return symbols;
    }
    fn output_symbols(symbols:&Vec<Vec<u8>>,name:&str) {
        let start = Instant::now();
        // Create folder
        if !Path::new("split").exists() {
            fs::create_dir("split").unwrap();
        }
        // If folder exists, empty it.
        let path = format!("split/{}",name);
        if Path::new(&path).exists() {
            fs::remove_dir_all(&path).unwrap();// Delete folder
        }
        fs::create_dir(&path).unwrap(); // Create folder

        for i in 0..symbols.len() {
            let symbol_image = ImageBuffer::<Luma<u8>,Vec<u8>>::from_raw(
                SCALE,SCALE,
                symbols[i].iter().map(|&x| {if x==1{255}else{0}}).collect()
            ).unwrap();

            let path = format!("split/{}/{}.png",name,i);
            
            symbol_image.save(path).unwrap();
        }
        //Export bounds
        println!("{} : Symbols output",time(start));
    }
    fn get_bounds(debug_out:bool,pixel_lists:&Vec<Vec<(usize,usize)>>) -> Vec<(((usize,usize),(usize,usize)),((i32,i32),(i32,i32),i32))> {
        let start = Instant::now();
        // Gets bounds
        let mut bounds:Vec<((usize,usize),(usize,usize))> = Vec::new();
        let mut sqr_bounds:Vec<((i32,i32),(i32,i32),i32)> = Vec::new();
        
        for symbol in pixel_lists {
            let (mut lower_x,mut lower_y) = symbol[0];
            let (mut upper_x,mut upper_y) = symbol[0];
            for i in 1..symbol.len() {
                let (x,y) = symbol[i];

                if x < lower_x { lower_x = x; }
                else if x > upper_x { upper_x = x; }
                
                if y < lower_y { lower_y = y; }
                else if y > upper_y { upper_y = y; }
            }

            // Gets square bounds centred on original bounds
            bounds.push(((lower_x,lower_y),(upper_x,upper_y)));
            sqr_bounds.push(square_indxs(lower_x,lower_y,upper_x,upper_y));
        }

        if debug_out { println!("{} : Bounds set",time(start)); };

        return izip!(bounds,sqr_bounds).collect();

        // talk about adding this
        fn square_indxs(lower_x:usize,lower_y:usize,upper_x:usize,upper_y:usize) -> ((i32,i32),(i32,i32),i32) {
            let (view_width,view_height) = (upper_x-lower_x,upper_y-lower_y);
        
            let dif:i32 = view_width as i32-view_height as i32;
            let dif_by_2 = dif/2;
            // If width > height
            if dif>0 {
                return (
                    (lower_x as i32,lower_y as i32-dif_by_2),
                    (upper_x as i32,upper_y as i32+dif_by_2),
                    dif_by_2
                );
            }
            // If width < height (if 0 has no affect)
            else {
                return (
                    (lower_x as i32+dif_by_2,lower_y as i32),
                    (upper_x as i32-dif_by_2,upper_y as i32),
                    dif_by_2
                );
            } 
        }
    }
    fn output_bounds(spacing:usize,colour:[u8;3],path:&str,name:&str,symbols:&Vec<u8>,bounds:&Vec<(((usize,usize),(usize,usize)),((i32,i32),(i32,i32),i32))>,width:usize,height:usize) {
        let start = Instant::now();

        let i32_sp = spacing as i32;
        let border_bounds:Vec<((i32,i32),(i32,i32))> = bounds.iter().map(
            |&(_,((min_x,min_y),(max_x,max_y),_))| ((min_x-i32_sp,min_y-i32_sp),(max_x+i32_sp,max_y+i32_sp))
        ).collect();

        // Copies image
        let mut border_img = image::open(format!("test_imgs/{}",path)).unwrap().into_rgb();
        for (x,y,pixel) in border_img.enumerate_pixels_mut() {
            let val = symbols[((y as usize)*width)+(x as usize)];
            *pixel = image::Rgb([val,val,val]);
        }

        let (width,height) = (width as i32, height as i32);
        // Sets borders
        let border_pixel = image::Rgb(colour); // Pixel to use as border
        for symbol in border_bounds.iter() {
            let ((min_x,min_y),(max_x,max_y)) = *symbol;
            //println!("({},{})->({},{}) | ({},{})",min_x,min_y,max_x,max_y,width,height);

            // Sets horizontal borders
            let max_x_indx = if max_x < width { max_x } else { width };
            let min_x_indx = if min_x < 0 { 0 } else { min_x };
            //println!("{} / {}",max_x_indx,min_x_indx);
            for i in min_x_indx..max_x_indx {
                if min_y >= 0 { *border_img.get_pixel_mut(i as u32,min_y as u32) = border_pixel; }
                if max_y < height { *border_img.get_pixel_mut(i as u32,max_y as u32) = border_pixel; }
            }
            // Sets vertical borders
            let max_y_indx = if max_y < height { max_y } else { height };
            let min_y_indx = if min_y < 0 { 0 } else { min_y };
            //println!("{} / {}",max_y_indx,min_y_indx);
            for i in min_y_indx..max_y_indx {
                if min_x >= 0 { *border_img.get_pixel_mut(min_x as u32,i as u32) = border_pixel; }
                if max_x < width { *border_img.get_pixel_mut(max_x as u32,i as u32) = border_pixel; }
            }
            // Sets bottom corner border
            if max_x < width && max_y < height {
                *border_img.get_pixel_mut(max_x as u32,max_y as u32) = border_pixel;
            }
           
        }
        if !Path::new("borders").exists() {
            fs::create_dir("borders").unwrap();
        }
        border_img.save(format!("borders/{}.png",name)).unwrap();

        println!("{} : Bounds output",time(start));
    }
    
}
fn construct(debug_out:bool,classes:&[&str],bounds:&Vec<((usize,usize),(usize,usize))>) -> String {
    // Struct 2p point coordinates
    #[derive(Copy,Clone,Debug)]
    struct Point {
        x:usize,
        y:usize,
    }
    impl Sub for Point {
        type Output = Self;
        fn sub(self,other:Self) -> Point {
            Self { 
                x:self.x-other.x,
                y:self.y-other.y,
            }
        }
    }
    impl SubAssign for Point {
        fn sub_assign(&mut self,other:Self) {
            *self = Self { 
                x:self.x-other.x,
                y:self.y-other.y,
            }
        }
    }
    // Struct for 2d bound
    #[derive(Clone,Debug)]
    struct Bound { min:Point,max:Point }
    impl SubAssign<Point> for Bound {
        fn sub_assign(&mut self,other:Point) {
            *self = Self { min:self.min - other, max:self.max - other }
        }
    }
    impl Bound {
        fn y_center(&self) -> usize {
            (self.min.y+self.max.y) / 2
        }
        // If x bounds of self contain all x bounds in `others`.
        fn contains_x(&self,others:&[&Bound]) -> bool {
            for bound in others {
                if bound.min.x < self.min.x || bound.max.x > self.max.x { return false; }
            }
            return true;
        }
    }
    // Constructs bound around given bounds
    impl From<&Vec<&Bound>> for Bound {
        fn from(bounds:&Vec<&Bound>) -> Self {
            Bound { 
                min:Point { 
                    x:bounds.iter().min_by_key(|p| p.min.x).unwrap().min.x, 
                    y:bounds.iter().min_by_key(|p| p.min.y).unwrap().min.y
                },
                max:Point { 
                    x:bounds.iter().max_by_key(|p| p.max.x).unwrap().max.x, 
                    y:bounds.iter().max_by_key(|p| p.max.y).unwrap().max.y
                },
            }
        }
    }
    // Struct for symbol
    #[derive(Clone,Debug)]
    struct Symbol {
        class:String,
        bounds:Bound
    }
    // Struct for row
    #[derive(Debug)]
    struct Row {
        center:usize,
        height:usize,
        sum:usize,
        symbols:Vec<Symbol>,
        superscript:Option<*mut Row>,
        subscript: Option<*mut Row>,
        parent:Option<*mut Row>,
    }
    impl Row {
        fn print_symbols(&self) -> String {
            format!("[{}]",self.symbols.iter().map(|s| format!("{} ",s.class)).collect::<String>())
        }
    }
    
    let start = Instant::now();

    // Converts given symbols and bounds into `Symbol` structs
    let mut combined:Vec<Symbol> = izip!(classes.iter(),bounds.iter())
        .map(|(&class,bounds)| 
            Symbol {
                class: class.to_string(),
                bounds: Bound { min:Point{ x:(bounds.0).0, y:(bounds.0).1}, max:Point{ x:(bounds.1).0, y:(bounds.1).1}}
            }
    ).collect();

    // Sorts symbols by min x bound, ordering symbols horizontally
    // O(n log n)
    combined.sort_by_key(|a| ((a.bounds).min).x);

    // min x and y out of all symbols
    let min_x:usize = combined[0].bounds.min.x; // O(1)
    let min_y:usize = (bounds.iter().min_by_key(|(min,_)| min.1).expect("Bounds empty").0).1; // O(n)
    let origin = Point { x:min_x, y:min_y};
    
    // Subtract mins (`origin`) from bounds of all symbols
    for row in combined.iter_mut() {
        row.bounds -= origin;
    }

    // Calculates center y coord of each symbol
    let y_centers:Vec<usize> = combined.iter().map(|s| s.bounds.y_center()).collect();

    // Initialises rows, 1st row containing 1st symbol
    let mut rows:Vec<Row> = vec![Row{
        center:y_centers[0],
        height:usize::default(),
        sum:y_centers[0],
        symbols:vec![combined[0].clone()],
        superscript:None,
        subscript:None,
        parent:None
    }];
    
    // Iterator skips 1st symbol
    let mut iter = izip!(y_centers,combined);
    iter.next();
    // Iterates across symbols and their centers
    for (y_center,symbol) in iter {
        let mut new_row = true;
        // Iterate through existing rows checking if this symbols belongs to one
        for t in 0..rows.len() {
            // If center of symbol is less than x% different, then it belongs to row. (x=100*ROW_CLEARANCE)
            if (1f32-(y_center as f32 / rows[t].center as f32)).abs() < ROW_CLEARANCE {
                rows[t].symbols.push(symbol.clone());
                rows[t].sum += y_center;
                rows[t].center = rows[t].sum / rows[t].symbols.len();
                new_row = false; // Identifies a new row is not needed to contain said symbol
                break;
            }
        }
        // If symbol not put in existing row, create a new one.
        if new_row {
            rows.push(Row{
                center:y_center,
                height:usize::default(),
                sum:y_center,
                symbols:vec![symbol.clone()],
                superscript:None,
                subscript:None,
                parent: None,
            });
        }
    }

    // Prints symbols in rows
    if debug_out {
        println!("rows (base):");
        for (indx,row) in rows.iter().enumerate() {
            println!("\t{} : {}",indx,row.print_symbols());
        }
    }
    

    // Construct composite symbols
    for row in rows.iter_mut() {
        let mut i = 0usize;
        // Can't use for loop since we use `.remove()` in loop (TODO Double check this)
        while i < row.symbols.len() {
            if row.symbols[i].class == "-" {
                if i+1 < row.symbols.len() {
                    if row.symbols[i+1].class == "-" {
                        // If difference between min x's is less than 20%
                        if (1f32 - row.symbols[i].bounds.min.x as f32 / row.symbols[i+1].bounds.min.x as f32).abs() <= 0.2f32 {
                            // Sets new symbol 
                            row.symbols[i].class="=".to_string();  // `=`
                            // Sets bounds
                            row.symbols[i].bounds = Bound::from(&vec![&row.symbols[i].bounds,&row.symbols[i+1].bounds]); // TODO How could I use slices here?
                            // Removes component part
                            row.symbols.remove(i+1);
                        }
                    }
                    else if i+2 < row.symbols.len() {
                        // If `row.symbols[i+1]` and `row.symbols[i+2]` are contained within `row.symbols[i]`
                        if
                        row.symbols[i+1].class == "\\cdot " && row.symbols[i+2].class == "\\cdot " && 
                            row.symbols[i].bounds.contains_x(&[&row.symbols[i+1].bounds,&row.symbols[i+2].bounds]) 
                        {
                            // Sets new symbol 
                            row.symbols[i].class = "\\div ".to_string(); // `\div`
    
                            // Calculate y bounds (which "." is on top and which is on bottom)
                            let (min_y,max_y) = if row.symbols[i+1].bounds.min.y < row.symbols[i+2].bounds.min.y {
                                (row.symbols[i+1].bounds.min.y,row.symbols[i+2].bounds.max.y)
                            } else {
                                (row.symbols[i+2].bounds.min.y,row.symbols[i+1].bounds.max.y)
                            };
                            // Sets bounds
                            row.symbols[i].bounds = Bound {
                                min:Point{ x:row.symbols[i+1].bounds.min.x, y:min_y},
                                max:Point{ x:row.symbols[i+1].bounds.max.x, y:max_y}
                            };
                            // Removes component part
                            row.symbols.remove(i+1);
                            row.symbols.remove(i+1); // After first remove now i+1 == prev i+2
                        }
                    }
                }
            }
            i += 1;
        }
    }

    // Prints symbols in rows
    if debug_out {
        println!("rows (combined symbols):");
        for (indx,row) in rows.iter().enumerate() {
            println!("\t{} : {}",indx,row.print_symbols());
        }
    }
    // Sorts rows in vertical order rows[0] is top row
    rows.sort_by_key(|r| r.center);

    // Prints symbols in rows and row centers
    if debug_out {
        println!("rows (vertically ordered):");
        for (indx,row) in rows.iter().enumerate() {
            println!("\t{} : {}",indx,row.print_symbols());
        }
        let centers:Vec<usize> = rows.iter().map(|x|x.center).collect();
        println!("row centers: {:.?}",centers);
    }

    // Calculates average height of rows
    for row in rows.iter_mut() {
        let mut ignored_symbols = 0usize;
        for symbol in row.symbols.iter() {
            // Ignore the heights of '-' and '\\cdot' since there minuscule heights will throw off the average
            match symbol.class.as_str() {
                "-" | "\\cdot" => ignored_symbols +=1,
                _ => row.height += symbol.bounds.max.y - symbol.bounds.min.y, // `symbol.bounds.1.y - symbol.bounds.0.y` = height of symbol
            }
        }
        // Average height in row
        if row.symbols.len() != ignored_symbols {
            row.height /= row.symbols.len()-ignored_symbols;
        }
    }

    // Prints average row heights
    if debug_out {
        let heights:Vec<usize> = rows.iter().map(|x|x.height).collect();
        println!("row heights: {:.?}",heights);
    }

    // Contains references to rows not linked to another row as a sub/super script row
    // Initially contains a reference to every row.
    let mut unassigned_rows:Vec<&mut Row> = rows.iter_mut().map(|x|x).collect();

    // Only 1 row is not a sub/super script row of another.
    // When we only have 1 unreferenced row we know we have linked all other rows as sub/super scripts.
    while unassigned_rows.len() > 1 {
        // List of indexes in reference to rows to remove from unassigned_rows as they have been assigned
        let mut removal_list:Vec<usize> = Vec::new();
        for i in 0..unassigned_rows.len() {

            let mut pos_sub = false; // Defines if this row could be a subscript row.
            if i > 0 { // If there is a row above this.
                // If the height of the row above is more than this, this could be a subscript to the row below
                if unassigned_rows[i-1].height> unassigned_rows[i].height  {
                    pos_sub = true;
                }
            }

            let mut pos_sup = false; // Defines if this row could be a superscript row.
            if i < unassigned_rows.len()-1 { // If there is a row below this.
                // If the height of the row below is more than this, this could be a superscrit to the row below.
                if unassigned_rows[i+1].height > unassigned_rows[i].height  {
                    pos_sup = true;
                }
            }

            // Gets mutable raw pointer to this row
            let pointer:*mut Row = *unassigned_rows.get_mut(i).unwrap() as *mut Row;
            // If could both be superscript and subscript.
            // This row is a sub/super script to the row with smallest height
            if pos_sup && pos_sub {
                // If row below is smaller than row above, this row is a superscript to row below
                if unassigned_rows[i+1].height < unassigned_rows[i-1].height {
                    unassigned_rows[i+1].superscript = Some(pointer); // Links parent to this as subscript
                    unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i+1).unwrap() as *mut Row); // Links to parent
                    removal_list.push(i);
                }
                // Else it is the subscript to the row above
                else {
                    unassigned_rows[i-1].subscript = Some(pointer); // Links parent to this as superscript
                    unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i-1).unwrap() as *mut Row); // Links to parent
                    removal_list.push(i);
                }
            }
            // If could only be superscript
            else if pos_sub {
                unassigned_rows[i-1].subscript = Some(pointer); // Links parent to this as superscript
                unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i-1).unwrap() as *mut Row); // Links to parent
                removal_list.push(i);
            }
            // If could only be subscript
            else if pos_sup {
                unassigned_rows[i+1].superscript = Some(pointer); // Links parent to this as subscript
                unassigned_rows[i].parent = Some(*unassigned_rows.get_mut(i+1).unwrap() as *mut Row); // Links to parent
                removal_list.push(i);
            }
        }
        // Removes assigned rows from `unassigned_rows`
        remove_indexes(&mut unassigned_rows, &removal_list);
    }
    //println!("finished script setting");

    // Prints rows and linked rows (doesn't use `debug_out` since output is large, complex and interferes with later code. Not good for an overview)
    // unsafe {
    //     println!("\nrows:");
    //     for row in rows.iter() {
    //         println!();
    //         println!("{:.?}",row.print_symbols());
    //         if let Some(pointer) = row.subscript {
    //             println!("sub: {:.?} -> {:.?}",pointer,(*pointer).print_symbols())
    //         }
    //         if let Some(pointer) = row.superscript {
    //             println!("sup: {:.?} -> {:.?}",pointer,(*pointer).print_symbols())
    //         }
    //     }
    // }
    // return "".to_string();

    if debug_out { println!("{} : Scripts set",time(start)); }

    // Sets 1st row
    let mut current_row:&mut Row = unassigned_rows.get_mut(0).unwrap();
    // Sets 1st symbol in latex
    let mut latex:String = String::from(format!("{}",current_row.symbols[0].class));
    // Removes set symbol from row
    current_row.symbols.remove(0);
    unsafe {
        loop  {
            if debug_out { println!("building: {}",latex); }

            // TODO Make `min_sub` and `min_sup` immutable
            // Gets min x coordinate of next symbol in possible rows.
            //----------
            // Gets min x bound of symbol in subscript row
            let mut min_sub:usize = usize::max_value();
            if let Some(sub_row) = current_row.subscript {
                if let Some(symbol) = (*sub_row).symbols.first() {
                    min_sub = symbol.bounds.min.x;
                }
            }
            // Gets min x bound of symbol in superscript row
            let mut min_sup:usize = usize::max_value();
            if let Some(sup_row) = current_row.superscript {
                if let Some(symbol) = (*sup_row).symbols.first() {
                    min_sup = symbol.bounds.min.x;
                }
            }
            // Gets min x bound of next symbol in current row
            let min_cur:usize = if let Some(symbol) = current_row.symbols.get(0) {
                symbol.bounds.min.x
            } else { usize::max_value() };

            // Gets min x bounds of symbol in parent row
            let mut min_par:usize = usize::max_value();
            if let Some(parent) = current_row.parent {
                if let Some(symbol) = (*parent).symbols.first() {
                    min_par = symbol.bounds.min.x;
                }
            }
    
            //println!("(sub,sup,cur,par):({},{},{},{})",min_sub,min_sup,min_cur,min_par);

            // Finds minimum min x coordinate of next symbol in possible rows,
            //  switches to that row, appeneds that symbol to `latex` and removes that symbol from its row.
            //----------
            // ('closest' in this section means horizontally closest)
            if let Some(min) = min_option(&[min_sub,min_sup,min_cur,min_par]) {
                // If next closest symbol resides in the parent row, close this row and swtich to the parent row.
                if min == min_par {
                    current_row = &mut *current_row.parent.unwrap();
                    latex.push_str("}");
                }
                // If next closest symbol does not resides in the parent row
                else {
                    // If next closest symbol resides in subscript row, open subscript row, push 1st symbol and switch row.
                    if min == min_sub {
                        current_row = &mut *current_row.subscript.unwrap();
                        latex.push_str(&format!("_{{{}",current_row.symbols[0].class));
                    }
                    // If next closest symbol resides in superscript row, open subscript row, push 1st symbol and switch row.
                    else if min == min_sup {
                        current_row = &mut *current_row.superscript.unwrap();
                        latex.push_str(&format!("^{{{}",current_row.symbols[0].class));     
                    }
                    // If next closest symbol resides in current row, push next symbol in current row.
                    else if min == min_cur {
                        latex.push_str(&current_row.symbols[0].class);
                    }
                    // Remove symbol added to latex
                    current_row.symbols.remove(0);
                }
            }
            // If next closest symbol not in parent, current, subscript or superscript row, 
            else {
                // If there exists a parent row, close row and switch to parent row.
                if let Some(parent) = current_row.parent {
                    current_row = &mut *parent;
                    latex.push_str("}");
                }
                // If there does not exist a parent row, we are in the base row and at the end of the equation.
                else {
                    break;
                }
                
            }
        }
    }

    if debug_out { println!("{} : Construction finished",time(start)); }

    return latex;

    // Returns `Some(min)` (min=minimum value) from 4 element usize list unless minimum value equals `usize::max_value`, in which case it returns `None`.
    fn min_option(slice:&[usize;4]) -> Option<usize> {
        let min = *slice.iter().min().unwrap();
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

    fn remove_indexes<T>(vec:&mut Vec<T>,indxs:&[usize]) {
        let mut counteracter = 0usize;
        for indx in indxs {
            vec.remove(indx-counteracter);
            counteracter += 1;
        }
    }
}

#[allow(dead_code)]
fn time(instant:Instant) -> String {
    let mut millis = instant.elapsed().as_millis();
    let seconds = (millis as f32 / 1000f32).floor();
    millis = millis % 1000;
    let time = format!("{:#02}:{:#03}",seconds,millis);
    return time;
}