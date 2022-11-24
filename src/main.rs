extern crate csv;
use linfa::{traits::Fit, Dataset};
use linfa_trees::{DecisionTree, SplitQuality};
use ndarray::{s, Array2, Axis};
use ndarray_csv::Array2Reader;
use std::fs::File;
use std::io::Write;

fn main() {
    let mut reader = csv::Reader::from_path("happiness_tree.csv").unwrap();
    let csv_data: Array2<String> = reader.deserialize_array2_dynamic().unwrap();
    // Turn the strings from the csv data into f32 values
    let data = csv_data.mapv(|elem| elem.parse::<f32>().unwrap());

    let num_features = data.len_of(Axis(1)) - 1;

    // Get the headers into a vec
    let mut feature_headers: Vec<String> = Vec::new();
    for element in reader.headers().unwrap().into_iter() {
        feature_headers.push(String::from(element))
    }
    // We don't want to use the last header column; that's going to be turned
    // into a label later
    let feature_names = feature_headers[0..num_features].to_vec();

    // We don't want to include the last data column in our features; that's
    // going to be used as our label data
    let features = data.slice(s![.., 0..num_features]).to_owned();

    let labels = data.column(num_features).to_owned();

    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match x.to_owned() as i32 {
            i32::MIN..=4 => "Sad",
            5..=7 => "Ok",
            8..=i32::MAX => "Happy",
        })
        .with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();

    File::create("./model_output/dt.tex")
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();
}
