let colors = [];
let POINT_OUTER_WEIGHT = 15;
let POINT_WEIGHT = 10;

let C = 100;
let alpha = 0.002;
let COST_THRESH = 0.001;
//  [-0.0188778656858697, -0.029626756811370558, -0.0795166809900494]
// let W = [Math.random()*600 - 300, 1, 1];
let W = [0, 1, -1];
let BestW = W, BestLoss;

let stochasticIndex = 0;
let stochasticDescent = false;
let iterations = 0;

let dataX = [
    // [261, -87], [188, -128], [117, -178], [87, -201], [146, -217],
    // [217, -190], [255, -136], [281, -127], [305, -137], [308, -196],
    // [234, -214], [197, -227], [-26, -132], [-82, -131], [-16, -61],
    // [80, -13], [138, 33], [190, 80], [201, 119], [148, 126], [40, 102]
];
let dataY = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1];
let points = [];

let Min = [], Max = [];

for (let i = 0; i < dataX.length; i++) {
    points.push({
        x: dataX[i],
        label: dataY[i]
    });
}

let train = false;
let error_hist = [];


function svmCost(W, xs, real) {
    let hinge_loss = 0.0;
    let N = xs.length;

    // Compute Hinge Loss
    for (let i = 0; i < N; i++) {
        hinge_loss += Math.max(0, 1 - real[i]*score(W, xs[i]));
    }
    hinge_loss = C * (hinge_loss / N);

    // Add to Cost SVM Margin Minimization
    let error = hinge_loss + (1.0/2.0)*mag2(W);

    return error;
}

function update(W, xs, real) {
    let cW = [...W];
    let N = xs.length;

    for (let j = 0; j < cW.length; j++) {
        // Compute SubGrad for Parameter j
        let grad = 0.0;
        for (let i = 0; i < N; i++) {
            if (Math.max(0, 1 - real[i] * score(cW, xs[i])) == 0) {
                grad += cW[j];
            }
            else {
                grad += cW[j] - C*real[i]*xs[i][j];
            }
        }
        grad /= N;

        // Update Parameter j
        cW[j] = cW[j] - alpha*grad;
    }

    return cW;
}

function wToStr(W) {
    return `${W[1].toFixed(4)}x + ${W[2].toFixed(4)}y - ${W[0].toFixed(4)}`;
}

function sMiniMax(x, min, max) {
    return (x = min) / (max - min);
}

function iMiniMax(ux, min, max) {
    return ux * (max - min) + min;
}

function computeMinMax(points) {
    let xs = points.map(p => p.x);
    Min = [xs[0][0], xs[0][1]];
    Max = [xs[0][0], xs[0][1]];
    for (let i = 0; i < xs.length; i++) {
        for (let j = 0; j < xs[i].length; j++) {
            if (xs[i][j] < Min[j])
                Min[j] = xs[i][j];
            if (xs[i][j] > Max[j])
                Max[j] = xs[i][j];
        }
    }
}

function normInstance(x) { 
    return [x[0], sMiniMax(x[1], Min[0], Max[0]), sMiniMax(x[2], Min[1], Max[1])];
}

function normXs(xs) {
    return xs.map(x => {
        return normInstance(x);
    });
}

function score(W, x) {
    let score = 0;
    for (let j = 0; j < W.length; j++) {
        score += x[j] * W[j];
    }

    return score;
}

// function score(x) {
//     let score = 0;
    
//     let diff = 0;
//     for (let j = 0; j < W.length; j++) {
//         diff += Math.pow(x[j] - W[j], 2);
//     }

//     Math.exp(-1 * diff);

//     return score;
// }


function mag(W) {
    return Math.sqrt(W[0]*W[0] + W[1]*W[1]);
}

function mag2(W) {
    let magn = 0.0;
    for (let w of W) {
        magn += w*w;
    }
    return magn;
}


function cToInd(label) {
    return label == 1 ? 0 : 1;
}
function T(point) {
    return [point[0] - width/2, 
            -1*point[1] + height/2];
}

function TInv(point) {

}

function setup() {
    createCanvas(640, 480);
    colors = [
        color(255, 0, 0),
        color(0, 255, 0),
        color(0, 0, 255)
    ];

    shuffleArray(points);

    C_slider = createSlider(0, 100, C, 0);
    C_slider.position(10,60);
    C_slider.style('width', '80px');

    ResetToBest_button = createButton('Set to Best');
    ResetToBest_button.position(10, 170);
    ResetToBest_button.mousePressed(() => {
        W = BestW;
    });

    Stochastic_checkbox = createCheckbox('Stochastic Descent');
    Stochastic_checkbox.position(10, 10);
    Stochastic_checkbox.changed(() => {
        stochasticDescent = Stochastic_checkbox.checked();
    })

}


function draw() {
    fill(255);
    strokeWeight(1);
    rect(0,0, width, height);
    stroke(0);
    fill(0);

    if (error_hist.length > 0)
        text("Cost: " + error_hist[error_hist.length-1].toFixed(4), 10, 10);

    text("Iter #: " + iterations, 10, 30);

    C = C_slider.value();
    text("C: " + C, 10, 50);

    text("Curr W:", 10, 100);
    text(wToStr(W), 10, 110);

    text("Best W:", 10, 130);
    text(wToStr(BestW), 10, 140);
    
    translate(width/2, height/2);
    scale(1, -1);

    line(-width/2, 0, width/2, 0);
    line(0, -height/2, 0, height/2);

    drawDecisionBoundary(BestW);
    
    for (let p of points) {
        strokeWeight(POINT_OUTER_WEIGHT);
        if (score(BestW, [-1, ...p.x]) >= 0) {
            stroke(0, 0, 0);
        } else {
            stroke(0, 255, 255);
        }
        point(p.x[0], p.x[1]);

        let lInd = cToInd(p.label);
        fill(colors[lInd]);
        stroke(colors[lInd]);
        strokeWeight(POINT_WEIGHT);
        point(p.x[0], p.x[1]);
    }

    

    if (train) {


        shuffleArray(points);

        let ys = points.map(p => p.label);
        // let xs = points.map(p => [-1, p.x[0]/(width/2), p.x[1]/(height/2)]);
        let xs = points.map(p => [-1, ...p.x]);
        
        if (stochasticDescent) {
            xs = xs.slice(stochasticIndex, stochasticIndex+1);
            ys = ys.slice(stochasticIndex, stochasticIndex+1);
        }
        
        W = update(W, xs, ys);

        if (stochasticDescent) {
            stochasticIndex = stochasticIndex + 1;
            if (stochasticIndex == points.length) {
                stochasticIndex = 0;
                iterations += 1;
            }
        } else {
            iterations += 1;
        }

        let loss = svmCost(W, xs, ys);
        error_hist.push(loss);

        if (loss < BestLoss) {
            BestLoss = loss;
            BestW = W;
        }

        let errN = error_hist.length;
        if (errN > 2) {
            ERR_TOL = 0.00001;
            if (Math.abs(error_hist[errN-1] - error_hist[errN-2]) < ERR_TOL) {
                train = false;
                console.log("Min Error: " + error_hist[errN-1]);
                console.log("Trained!");
            }
        }
    }

}

function drawDecisionBoundary(W) {
    let bx1 = -width/2,
        bx2 = width/2;

    let ux1 = sMiniMax(bx1, Min[0], Max[0]),
        ux2 = sMiniMax(bx2, Min[0], Max[0]);
    
    let by1 = (-W[1]*bx1 + W[0]) / W[2],
        by2 = (-W[1]*bx2 + W[0]) / W[2];

    // let by1 = iMiniMax(uy1, Min[1], Max[1]),
    //     by2 = iMiniMax(uy2, Min[1], Max[1]);

    let v1 = new p5.Vector(bx1, by1),
        v2 = new p5.Vector(bx2, by2);

    stroke(0,0,255);
    line(v1.x, v1.y, v2.x, v2.y);

    let o = new p5.Vector(0,0);
    let w = new p5.Vector(W[1], W[2]);
    w = w.mult(50);
    stroke(255, 0, 255);
    strokeWeight(3);
    line(o.x, o.y, w.x, w.y);

    // let dirVector = v2.add(v1.mult(-1));
    // dirVector = dirVector.normalize();
    // let halfDist = v1.dist(v2) / 2;

    // let halfPoint = v1.add(dirVector.mult(halfDist));

    // stroke(255, 0, 255);
    // strokeWeight(20);
    // point(halfPoint.x, halfPoint.y);

    for (let x = -width/2; x <= width/2; x += 20) {
        strokeWeight(10);
        stroke(0);
        point(x, (-W[1]*x + W[0]) / W[2])
    }
}

function keyPressed() {
    if (key == 'p') {
        points.push({
            x: T([mouseX, mouseY]),
            label: 1
        });

        computeMinMax(points);
    }
    else if (key == 'q') {
        points.push({
            x: T([mouseX, mouseY]),
            label: -1
        });

        computeMinMax(points);
    }

    else if (keyCode == ENTER) {

        if (!train) {
            // TRAIN!
            computeMinMax(points);
            let ys = points.map(p => p.label);
            let xs = points.map(p => [1, ...p.x]);
    
            let initialLoss = svmCost(W, xs, ys);
            console.log("Error: " + initialLoss);
            BestLoss = initialLoss;
            train = true;
        }
        else {
            train = false;
        }
    }


}


const shuffleArray = array => {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      const temp = array[i];
      array[i] = array[j];
      array[j] = temp;
    }
  }