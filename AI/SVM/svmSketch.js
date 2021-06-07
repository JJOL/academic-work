/**
 * Author: Juan Jo Olivera
 * Name: LinearSVM Java Script Implementation
 * Description: A simple graphical implementation of a Linear SVM on 2 dimensional points.
 * The cost function is the regularization strength parametized cost and the optimizer used
 * is Subgradient descent with fixed step size.
 * 
 */

let VISUALS;

let points = [];

let currSVM = new SVMModel();
let bestSVM = new SVMModel(currSVM);

let optimizer = new SVMOptimizer(currSVM, bestSVM, points, updateErrorGraph);

let training = false;

// Hyper Parameters
let lr = 0.002;
let C = 100;
let MAX_ITERS = 500;
let ERR_THRESHOLD = 0.0001;


let stochasticLearning = false;


// UI Controls
let rstBtn;
let cInCtl;
let lrInCtl;
let stochCheckBox;
let restoreBestBtn;
let resetLearningBtn;
let maxIterInCtl;
let errThreshrInCtl;

let hingeChart, marginChart, totalChart;

function setup() {
    var canvas = createCanvas(640, 640);
    canvas.parent('grapher-container');

    VISUALS = new PointsVisualization();

    rstBtn = createButton('Reset');
    rstBtn.position(15, 35);
    rstBtn.mousePressed(() => resetAll());
    cInCtl = createInput(C);
    cInCtl.position(85, 64);
    cInCtl.style('width', '40px');
    cInCtl.style('height', '10px');
    cInCtl.input(() => C = parseFloat(cInCtl.value()));
    lrInCtl = createInput(lr);
    lrInCtl.position(85, 78);
    lrInCtl.style('width', '40px');
    lrInCtl.style('height', '10px');
    lrInCtl.input(() => lr = parseFloat(lrInCtl.value()));
    stochCheckBox = createCheckbox('Stochastic');
    stochCheckBox.position(13, 95);
    stochCheckBox.changed(() => stochasticLearning = stochCheckBox.checked());
    restoreBestBtn = createButton('Set Best SVM');
    restoreBestBtn.position(15, 180);
    restoreBestBtn.mousePressed(() => restoreBestSVN());
    resetLearningBtn = createButton('Reset Learning');
    resetLearningBtn.position(15, 200);
    resetLearningBtn.mousePressed(() => resetLearning());
    maxIterInCtl = createInput(MAX_ITERS);
    maxIterInCtl.position(600, 585);
    maxIterInCtl.style('width', '40px');
    maxIterInCtl.style('height', '10px');
    maxIterInCtl.input(() => MAX_ITERS = parseFloat(maxIterInCtl.value()));
    errThreshrInCtl = createInput(ERR_THRESHOLD);
    errThreshrInCtl.position(600, 605);
    errThreshrInCtl.style('width', '40px');
    errThreshrInCtl.style('height', '10px');
    errThreshrInCtl.input(() => ERR_THRESHOLD = parseFloat(errThreshrInCtl.value()));

    createErrorGraphElements();
}

function draw() {

    // updateHyperParams();
    updateModel();

    drawGrid();
    drawPoints();
    drawDecisionBoundary();
    drawInfo();
}


function drawGrid() {
    fill(255);
    stroke(0);
    strokeWeight(1);
    rect(0, 0, width, height);

    let h1 = T.Out([-1, 0]),
        h2 = T.Out([1, 0]);
    line(h1[0], h1[1], h2[0], h2[1]);

    let v1 = T.Out([0, 1]),
        v2 = T.Out([0, -1]);
    line(v1[0], v1[1], v2[0], v2[1]);
}

function drawPoints() {
    for (let p of points) {
        let px = T.Out(p.x);

        let score = bestSVM.score(T.ext(p.x));

        strokeWeight(VISUALS.POINT_OUTER_WEIGHT);
        stroke(VISUALS.colorForScore(score));
        point(px[0], px[1]);
        
        stroke(VISUALS.colorForLabel(p.y));
        strokeWeight(VISUALS.POINT_WEIGHT);
        point(px[0], px[1]);
    }
}

function drawDecisionBoundary() {
    let bx1 = -1,
        bx2 = 1;
    
    let by1 = bestSVM.getYFromX(bx1),
        by2 = bestSVM.getYFromX(bx2);

    let v1 = T.Out([bx1, by1]),
        v2 = T.Out([bx2, by2]);

    strokeWeight(3);

    stroke(0,0,255);
    line(v1[0], v1[1], v2[0], v2[1]);

    stroke(255,0,255);
    let w = new p5.Vector(bestSVM.w[0], bestSVM.w[1])
    w = w.mult(0.2);
    w = T.Out([w.x, w.y]);
    let o = T.Out([0, 0]);
    line(o[0], o[1], w[0], w[1]);
}

function drawInfo() {
    stroke(0);
    strokeWeight(1);
    fill(0);

    text("N: " + points.length, 10, 20);
    text("+ " + points.filter(p => p.y == 1).length, 50, 20);
    text("- " + points.filter(p => p.y == -1).length, 80, 20);

    text("C: " + C, 10, 70);
    text("LR: " + lr, 10, 85);

    text("Curr W:", 10, 120);
    text(currSVM.toString(), 10, 133);
    
    text("Best W:", 10, 150);
    text(bestSVM.toString(), 10, 163);

    text("Iter #"+optimizer.iter, 10, 250);
    text("Max Iter: ", 520, 590);
    text("Error Thresh: ", 520, 610);
    if (optimizer.lastCost)
        text("Cost: " + optimizer.lastCost.toFixed(4), 10, 270);
}

function updateModel() {
    if (training) {
        optimizer.update();
    }
}

// Action Listeners

function keyPressed() {
    if (key == 'p') {
        addPoint(mouseX, mouseY, 1);
    }
    else if (key == 'q') {
        addPoint(mouseX, mouseY, -1);
    }
    else if (keyCode == ENTER) {
        toggleTraining();
    }
}

function resetAll() {
    points = [];

    currSVM = new SVMModel();
    bestSVM = new SVMModel(currSVM);
    optimizer = new SVMOptimizer(currSVM, bestSVM, points, updateErrorGraph);
}

function restoreBestSVN() {
    currSVM.set(bestSVM);
}

function resetLearning() {
    currSVM = new SVMModel();
    bestSVM = new SVMModel(currSVM);
    optimizer = new SVMOptimizer(currSVM, bestSVM, points, updateErrorGraph);
}

function addPoint(mx, my, label) {
    points.push({
        x: T.In([mx, my]),
        y: label
    });
}

function toggleTraining() {
    if (!training) {
        optimizer.beforeTrainStart();
        training = true;
    }
    else {
        training = false;
    }
}



function updateErrorGraph(totalLoss, hingeLoss, marginLoss) {

    let labels = [];
    for (let i = 0; i < hingeLoss.length; i++) {
        labels.push(i);
    }

    totalChart.data.labels = labels;
    totalChart.data.datasets.forEach(dataset => {
        dataset.data = totalLoss
    });
    totalChart.update();

    hingeChart.data.labels = labels;
    hingeChart.data.datasets.forEach(dataset => {
        dataset.data = hingeLoss
    });
    hingeChart.update();

    marginChart.data.labels = labels;
    marginChart.data.datasets.forEach(dataset => {
        dataset.data = marginLoss
    });
    marginChart.update();
}

function createErrorGraphElements() {
    let totalCtx = document.getElementById('total-loss-graph').getContext('2d');
    totalChart = new Chart(totalCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Total Loss',
                data: [],
                fill: false,
                borderColor: 'rgb(252, 186, 3)',
                tension: 0.1
            }]
        }
    });
    let hingeCtx = document.getElementById('hinge-loss-graph').getContext('2d');
    hingeChart = new Chart(hingeCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Hinge Loss',
                data: [],
                fill: false,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }]
        }
    });
    let marginCtx = document.getElementById('margin-loss-graph').getContext('2d');
    marginChart = new Chart(marginCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Margin Loss',
                data: [],
                fill: false,
                borderColor: 'rgb(255, 0, 255)',
                tension: 0.1
            }]
        }
    });
}