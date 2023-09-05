use std::f32::consts::E;

use math::{matrix, Matrix};

/// Node biases are included in weights as an extra row
/// Alpha is the 'learning rate', or how significantly the network changes with each iteration
#[derive(Debug)]
pub struct Network {
    pub layers: Vec<usize>,
    pub weights: Vec<Matrix>,
    pub alpha: f32,
}

impl Network {
    /// Creates a new neural network with the given layer
    /// architecture and training rate
    pub fn new(layers: &[usize], alpha: f32) -> Network {
        let mut weights = vec![];

        // Creates a weights matrix for every pair of layers except the last,
        // adding one extra column to represent the biases
        for i in 0..layers.len() - 2 {
            let w = matrix![layers[i] + 1; layers[i + 1] + 1];
            weights.push(w);
        }

        // Creates a weights matrix for the last two layers, without adding
        // an extra element at the end, since the output does not have a bias
        let w = matrix![layers[layers.len() - 2] + 1; layers[layers.len() - 1]];
        weights.push(w);

        Network {
            layers: layers.to_vec(),
            weights,
            alpha,
        }
    }

    /// Takes the sum of loss over all provided datasets, to
    /// measure how accurate the network's results are
    pub fn calculate_loss(&self, x_data: &[Matrix], y_data: &[Matrix]) -> f32 {
        let mut total = 0.0;
        for (input, target) in x_data.iter().zip(y_data) {
            let prediction = self.predict(input);
            total += 0.5 * (&prediction - &target).apply(|x| x.powf(2.0)).sum();
        }
        total
    }

    fn fit_partial(&mut self, input: Matrix, target: Matrix) {
        // Activations at each layer will be stored in this vector
        // The first layer's activation is just the input provided
        let mut activation = vec![input];

        // Feed forward step
        for layer_index in 0..self.weights.len() {
            let out = activation[layer_index]
                .multiply(&self.weights[layer_index])
                .apply(sigmoid);
            activation.push(out);
        }

        // Backpropagation step
        // The last layer of activation is the network's prediction
        let last_activation = &activation[activation.len() - 1];
        let error = last_activation - &target;

        let first_delta = &error * &last_activation.apply(sigmoid_deriv);
        let mut deltas = vec![first_delta];

        // Since Backpropagation works from the end to the beginning, we can
        // reverse the order of the for loop
        for layer_index in (1..activation.len() - 1).rev() {
            let previous_delta = &deltas[deltas.len() - 1];
            let weights = &self.weights[layer_index];

            let next_delta = previous_delta.multiply(&weights.transpose());
            let next_delta = &next_delta * &activation[layer_index].apply(sigmoid_deriv);
            deltas.push(next_delta);
        }

        // Finally, reverse the deltas since we created them in reverse order
        let deltas: Vec<Matrix> = deltas.into_iter().rev().collect();

        // Weight update step
        for layer_index in 0..self.weights.len() {
            // 'Alpha' is how significantly the weights change with each training item
            self.weights[layer_index] += activation[layer_index]
                .transpose()
                .multiply(&deltas[layer_index])
                .apply(|x| x * -self.alpha);
        }
    }

    pub fn fit(&mut self, x: &[Matrix], y_data: &[Matrix], epochs: usize, display_update: usize) {
        // println!(
        //     "[LOG] network layout: {}",
        //     self.layers
        //         .iter()
        //         .map(|x| x.to_string())
        //         .collect::<Vec<_>>()
        //         .join("-")
        // );
        // Insert a list of 1s as the last entry in the feature matrix
        // this allows us to treat the biases as another trainable parameter
        let mut x_data = vec![];
        for item in x {
            let mut new_item = item.clone();
            new_item.add_col(1.0);
            x_data.push(new_item);
        }

        for epoch in 0..epochs {
            // Repeat training for every item in the dataset
            for i in 0..x_data.len() {
                let input = x_data[i].clone();
                let target = y_data[i].clone();
                self.fit_partial(input, target);
            }
            // if epoch == 0 || (epoch + 1) % display_update == 0 {
            //     let loss = self.calculate_loss(&x_data, &y_data);
            //     println!("[INFO] epoch={}, loss={}", epoch, loss);
            // }
        }
        //println!("[LOG] training finished");
    }

    /// Provides a prediction from the network given a
    /// specific input
    pub fn predict(&self, x: &Matrix) -> Matrix {
        let mut input_data = x.clone();

        for layer_index in 0..self.weights.len() {
            input_data = input_data
                .multiply(&self.weights[layer_index])
                .apply(sigmoid);
        }

        input_data
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + E.powf(-x))
}

/// Calculates the derivative of sigmoid,
/// provided that x has already been passed through
/// the sigmoid function
fn sigmoid_deriv(x: f32) -> f32 {
    x * (1.0 - x)
}
