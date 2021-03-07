#[cfg(test)]
mod tests {
    use latex_classifier::{core::*, models::*};
    #[test]
    fn base() {
        // Runs segmentation
        // -----------------
        let segment = segment("tests/images/1.jpg");

        // Manually set correct classes for construction testing
        // -----------------
        let classes: [&str; 15] = [
            "3", "2", "x", "7", "1", "2", "\\cdot ", "b", "+", "-", "y", "-", "-", "-", "\\cdot ",
        ];

        // Gets bounds
        // -----------------
        let bounds: Vec<Bound<usize>> = segment.into_iter().map(|(_, b)| b).collect();

        assert!(false);

        // Gets LaTeX usually correctly set classes
        // -----------------
        let latex = construct(&classes, &bounds);

        // Prints LaTeX
        // -----------------
        println!("latex :{}", latex);
    }
}
