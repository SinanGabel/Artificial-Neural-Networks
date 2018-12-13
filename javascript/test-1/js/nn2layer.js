
"use strict";


// fathom.js library for AI experiments

// Note the difference between pointwise, and full array or matrix calculations

// check that all rows have same length

function isMatrix(m) {
    
    if (m[0] && m[0][0]) {
        
        return true;
    
    } else {
        
        return false;
    } 
}


/* 
  . rangeC(8, 3) = [8,8,8]
  . rangeC("a", 3) = ["a","a","a"]
*/  
function rangeC(c, n) {
  var r = [], i = 0;
  for(i = 0; i < n; i++) {
    r.push(c);
  }
  return r;
}


//    Marsaglia's polar method, http://en.wikipedia.org/wiki/Normal_distribution
//    1. Get sample y1, y2 from uniform distribution on (-1,1)
//    2. Accept if r = Math.pow(y1, 2) + Math.pow(y2, 2) < 1, else get new y1, y2 sample 
//    3. Define x1 = Sqrt[-2log(r)/r]y1, x2 = Sqrt[-2log(r)/r]y2
//    4. x1, x2 are two independent N(0,1) random variables
//    ToDo: add possibility to seed via numeric.seedrandom(x).random()
// 
// 
//    ToDo: ensure that s>0, n is an integer > 0
//    Note that 4 normal random numbers are generated per call.
//    m: mean, s: std. dev. = sqrt(variance)
// 
function normalxN(m, s, n) {

    var a, b;
    var c = b = 1;

    function normalx2() {
        c = b = 1;
        
        for (a = 2; 1 <= a;) {
            b = 2 * Math.random() - 1, c = 2 * Math.random() - 1, a = b * b + c * c;
        }
        return [Math.sqrt(-2 * Math.log(a) / a) * b, Math.sqrt(-2 * Math.log(a) / a) * c];
    }

  // ...
  if (0 > s || 1 > n) {

    console.log("normalxN(): parameters not correct: s : " + s + ", : n : " + n);

  } else {

    n = n || 10;
    let f = Math.ceil(n / 4), d = 0, g = [], e = [], h;

    if (0 === s || 0 === s) {
      for (d = 0;d < n;d++) {
        e.push(m);
      }
    } else {
      m = m || 0;
      for (s = s || 1;d < f;) {
        d++, h = normalx2(), g.push(h[0]), g.push(h[1]);
      }
      f = Math.ceil(n / 2);
      for (d = 0;d < f;d++) {
        e.push(m + s * g[d]), e.push(m - s * g[d]);
      }
      0 < 2 * f - n && e.pop();
    }
    return e;
  }
}


// --- Initialise ---

// method: standard normal, zero, ...

// ar: array of neural network layer dimensions e.g. 4 in input, 2 in a single hidden, 1 in output => ar = [4,2,1]
// ... thus 4 is the size of the input layer, 2 is the size of the hidden layer and 1 is the size of the output layer

// initialise("normal", [4,2,1]) or initialise("zero", [25,10,5,2])

// Possibly multiply with a factor to reduce initialisation weights i.e. numeric.mul(factor, initialise(method, ar))

// "In general, initializing all the weights to zero results in the network failing to break symmetry. This means that every neuron in each layer will learn the same thing, and you might as well be training a neural network with n[l]=1 for every layer, and the network is no more powerful than a linear classifier such as logistic regression.", source: Andrew Ng

function initialise(method, ar) {
    
    let res = [],
        l = 0, i = 0, j = 0;
    
    l = ar.length - 1;
    
    function fun(j, n2, n1) {
        
        let m = (method === "he") ? "normal" : method ;
        
        let out = {};
    
        if (m === "normal") {
            
            // initialise parameters with random numbers from a standard normal distribution x 0.01

            out = { "W": _.range(0, n2).map((c) => numeric.mul(0.01, normalxN(0, 1, n1))),
                     "b": rangeC(0, n2),
                     "layer": j
            }
        
        } else if (m === "zero") {
            
            out = { "W": _.range(0, n2).map((c) => rangeC(0, n1)),
                     "b": rangeC(0, n2),
                     "layer": j
            }
        }
        
        // xavier is 1/same
        
        if (method === "he") {
            
            out = numeric.mul( 2/Math.sqrt(n1), out);
            
        }
        
        return out;
    }
    
    for (i = 0; i < l; i++) {
        
        res.push(fun(i + 1, ar[i + 1], ar[i]));
    }
    
    return res;
}


// --- forward propagation ---

// Implement the forward propagation for the LINEAR->ACTIVATION layer
// 
//     Arguments:
//     A[L-1] -- activations from previous layer (or input data): (size of previous layer, number of examples)
//     W[L]   -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
//     b[L]   -- bias vector, numpy array of shape (size of the current layer, 1)
//     callback: activation -- relu, tanh, sigmoid

//     Returns:
//     A[L] = g[L](Z[L]) -- activations for this layer, also called the post-activation value

// linear_activation_forward(A,W,b,relu)

// Note: In deep learning, the "[LINEAR->ACTIVATION]" computation is counted as a single layer in the neural network, not two layers. 

// Example: var X = _.range(0, 10).map((c) => numeric.mul(0.01, normalxN(0, 1, 50))); var r = initialise("normal", [10,5,1]); var A= linear_activation_forward(X, r[0].W, r[0].b, relu); linear_activation_forward(A, r[1].W, r[1].b, relu)

function linear_activation_forward(A, W, b, callback) {
    
    // check dimensions, else transpose or warn    
    
    return numeric.dot(W, A).map((c,i) => callback(numeric.add(c, b[i])));
}




// --- ACTIVATION FUNCTIONS ---

// Why is a non-linear activation function needed? => A composition of two linear functions is still a linear function, so no matter how many layers there would be in the NN in essence it would be a linear activation function i.e. no need for hidden layers in that case.

// However with e.g. house price predictions in the output layer it could be okay to use a linear activation function.

// Note: Only use the sigmoid function for binary classification (output = 0 or 1) and only in the very last layer for yhat calculation 

// v: array of numbers

// returns ]0,1[

function sigmoid(v) {
    
    return v.map((c) => (1/(1 + Math.exp(-c))));
}


// Note: always better than sigmoid except in the case mentioned above

// returns ]-1,1[

function tanh(v) {
    
   return v.map((c) => (Math.tanh(c))); 
}


// ReLU: Rectified Linear Unit

// Note: normally use relu but else try the others too and see which works best

// returns [0,c]

function relu(v) {
    
   return v.map((c) => (Math.max(0, c))); 
}


// Note: normally just use the relu

function leaky_relu(v) {
    
   return v.map((c) => (Math.max(0.01 * c, c))); 
}


// --- First derivatives of ACTIVATION FUNCTIONS ---


// derivative of sigmoid(): _m1 first moment

// v: array of numbers

// s (optional): already calculated sigmoid

// Note: for logistic predictions it is normal to convert the outputs to 0 (if activation <= 0.5) and 1 (if activation > 0.5)

// Because we want our network to output probabilities the activation function for the output layer will be the softmax, which is simply a way to convert raw scores to probabilities. If you’re familiar with the logistic function you can think of softmax as its generalization to multiple classes.

function sigmoid_m1(v, s) {
    
    s = s || sigmoid(v);
    
    return s.map((c) => (c * (1 - c)));
}


// t (optional): already calculated tanh

function tanh_m1(v, t) {
    
    t = t || tanh(v);
    
    return t.map((c) => (1 - Math.pow(c,2)));
}


// Note: m1 in x=0, assume 0 or 1

function relu_m1(v) {
        
    return v.map((c) => (c > 0) ? 1 : 0 );
}


function leaky_relu_m1(v) {
        
    return v.map((c) => (c > 0) ? 1 : 0.01 );
}


// "Unroll" or reshape 3D image array (length, height, depth=3) to 1D array (length * height * depth, 1)

// ... e.g. (64,64,3) from image with 64x64 pixels and 3 primary colors: Red Green Blue (RGB)

// img: think of it as 2D-layout pixels in a 3-primary-color space

// Note: In course they convert to [[x1],[x2] ...] i.e. as a matrix => python shape (length * height * depth, 1)

function image2vector(img) {
    
    return _.flattenDepth(img, 2);
}


// normalizeRows divides each row vector of matrix m by its norm

// Note: images: if in RGB color then divide each color by 255 to normalise the image data

// Note: other methods: subtract mean and divide by standard deviation

function normalizeRows(m) {
    
    return m.map(r => numeric.div(r, numeric.norm2(r)));
}


// softmax is a normalizing function - summing to one (1.0000), or ones [1,1,1, ...] - used when needing to classify two or more classes.

// For C=2 i.e. two class, softmax reduces to linear regression (here note that if you have y_hat for one class the other is given by (1 - y_hat))

// if m is array return array, else return matrix

function softmax(m) {
    
    let exp = numeric.exp(m);
    
    if (isMatrix(m)) {
            
        return exp.map(r => numeric.div(r, numeric.sum(r)));
    
    } else {
        
        return numeric.div(exp, numeric.sum(exp));
    }
}

// L1 loss function: the sum of absolute differences between prediction values and true values (true in the sense of coming from supervised training examples)

// yhat array: predicted values
// y array   : true values

function lossL1(yhat, y) {
    
    let l1 = 0, i = 0,
        l = yhat.length;
    
    for (i = 0; i < l; i++) {
        
        l1 += Math.abs(y[i] - yhat[i]);
    }
    
    return l1;
}


// L2 loss function: the sum of squared differences between prediction values and true values (true in the sense of coming from supervised training examples)

// yhat array: predicted values
// y array   : true values

// python:     np.dot(y-yhat, y-yhat)
// javascript: ar = numeric.sub(y, yhat); numeric.dot(ar, ar)

function lossL2(yhat, y) {
    
    let l1 = 0, i = 0,
        l = yhat.length;
    
    for (i = 0; i < l; i++) {
        
        l1 += Math.pow(y[i] - yhat[i], 2);
    }
    
    return l1;
}


// Implement the cost function defined by equation (7).
// 
//     Arguments:
//     yhat = AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
//     Y         -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)
// 
//     Returns:
//     cost -- cross-entropy cost

// check that yhat and y have the same lengths

// example: cost_cross_entropy([0.33,0.33,0.33],[0, 1,1]) => 0.87, and cost_cross_entropy([0.33,0.66,0.66],[0, 1,1]) => 0.41

function cost_cross_entropy(yhat, y, w_array, lambda) {
    
    let m = 0, data_loss = 0, reg_loss = 0;
    
    m = y.length;  // number of (training / development / test) examples
    
    if (isMatrix(yhat)) {
        
        yhat = numeric.transpose(yhat);
    }

    data_loss = -1/m * numeric.sum( y.map((c,i) => (c * Math.log(yhat[i][c]) + (1 - c) * Math.log(1 - yhat[i][c]))) ) ;  // mean loss

//     data_loss = -1/m * numeric.sum( y.map((c,i) => (c * Math.log(yhat[i]) + (1 - c) * Math.log(1 - yhat[i]))) ) ;  // mean loss

//     data_loss = -1/m * numeric.sum(yhat.map((c,i) => (numeric.mul(y[i], numeric.log(c)) + numeric.mul(1 - y[i], numeric.log(numeric.sub(1, c)))))) ;  // mean loss
    
    reg_loss = (w_array) ? (1/(2*m)) * lambda * numeric.sum( w_array.map(w => numeric.norm2Squared(w)) ) : 0 ;  // regularization loss
    
    return data_loss + reg_loss;  // a number
}


// --- 2 layer neural network ---

// Call as: neuralnetwork2L(numeric.transpose(X2), Y2, 50, 10000, 0.05, 0.01)

// "Note that regularization hurts training set performance! This is because it limits the ability of the network to overfit to the training set." source: Andrew Ng

function neuralnetwork2L(X, Y, hidden_units, iterations, epsilon, lambda) {

    hidden_units = hidden_units || 25;  // hidden layer dimensionality i.e. the number of hidden units in the first hidden layer (the second hidden layer is the output layer)
    iterations = iterations || 2000;       
    epsilon = epsilon || 0.01;          // learning rate for gradient descent
    lambda = lambda || 0.01; // regularization strength
    
    var i = 0,
        m = X[0].length, // training set size

        n_x = 2, // input layer dimensionality
        n_h = hidden_units,  // hidden layer dimensionality
        n_y = 2, // output layer dimensionality
        
        one_hot = [];


    // initialise

    var init = initialise("he", [n_x, n_h, n_y]);

    var W1 = init[0].W;  // (3 x 2)
    var b1 = init[0].b;  // (3)

    var W2 = init[1].W;  // (2 x 3)
    var b2 = init[1].b;  // (2)
    
    // one_hot; reshape Y
    
    Y.forEach((c) => { one_hot.push( (c === 0) ? [1,0] : [0,1] ) });  // (200 x 2)
    
    one_hot = numeric.transpose(one_hot);  // (2 x 200)

    // forward propagation
    var A1, Z2, A2, dZ1, dZ2, dW1, dW2, db1, db2;

    for (i = 0; i < iterations; i++) {
    
        A1 = linear_activation_forward(X, W1, b1, tanh);  // (3 x 200)

        Z2 = linear_activation_forward(A1, W2, b2, (c) => (c));  // (3 x 200)

        A2 = numeric.transpose( numeric.transpose(Z2).map(c => softmax(c)) );  // (2 x 200)  probabilities == A2


        // calculate loss
        if ((i+1) % 1000 === 0) {

            console.log("cost: ", cost_cross_entropy(A2, Y, [W1, W2], lambda));  // note: reduces to: -c * Math.log(yhat[i][c]) because the case of c=0 yhat=0 and Math.log(1)=0 i.e. the second term of the cost cross entropy is zero.
        }
        
        // Back propagation
        
        dZ2 = numeric.sub(A2, one_hot);  // (2 x 200)
        
        
        dW2 = numeric.dot(dZ2, numeric.transpose(A1));  //  (2 x 200) x (200 x 3)  => (2 x 3)
        
        // add regularization gradient contribution
        numeric.addeq(dW2, numeric.mul(lambda, W2));
        
        dW2 = numeric.mul(1/m, dW2);
        
        db2 = dZ2.map((c) => (1/m * numeric.sum(c)));  // sum(2 x 200) => (2)
        
        // 2. backprop tanh
        
        
        dZ1 = numeric.dot(numeric.transpose(W2), dZ2)  // (3 x 2) x (2 x 200) => (3 x 200)
        
        dZ1 = numeric.mul(dZ1, numeric.sub(1, numeric.pow(A1,2)));
        
        // finally into W,b
        
        dW1 = numeric.dot(dZ1, numeric.transpose(X));  //  (3 x 200) x (200 x 2)  => (3 x 2)
        
        // add regularization gradient contribution
        numeric.addeq(dW1, numeric.mul(lambda, W1));

        dW1 = numeric.mul(1/m, dW1);

        db1 = dZ1.map((c) => (1/m * numeric.sum(c)));  // sum(3 x 200) => (3)
            
        // perform a parameter update
        
        numeric.subeq(W1, numeric.mul(epsilon, dW1));
        numeric.subeq(b1, numeric.mul(epsilon, db1));
        
        numeric.subeq(W2, numeric.mul(epsilon, dW2));
        numeric.subeq(b2, numeric.mul(epsilon, db2));    

    }
}

