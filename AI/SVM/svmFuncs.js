// Space Transformation Functions
class Transform {
    T1(p) {
        return [
            p[0] - (width/2),
            -p[1] + (height/2)
        ];
    }

    T1i(p) {
        return [
            p[0] + (width/2),
            -(p[1] - (height/2))
        ];
    }

    T2(p) {
        return [
            p[0] / (width/2),
            p[1] / (height/2)
        ];
    }
    
    T2i(p) {
        return [
            p[0] * (width/2),
            p[1] * (height/2)
        ];
    }

    In(p) {
        return this.T2(this.T1(p));
    }

    Out(p) {
        return this.T1i(this.T2i(p));
    }

    ext(p) {
        return [-1, ...p];
    }
}
const T = new Transform();
const shuffleArray = array => {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        const temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

class PointsVisualization {
    constructor() {
        this.trueColors = [
            color(255, 0, 0),
            color(0, 255, 0)
        ];

        this.scoreColors = [
            color(0, 0, 0),
            color(0, 255, 255)
        ];
    }
    colorForLabel(label) {
        return label == 1 ? this.trueColors[0] : this.trueColors[1];
    }

    colorForScore(score) {
        return score > 0 ? this.scoreColors[0] : this.scoreColors[1];
    }

    get POINT_WEIGHT() {
        return 10;
    }

    get POINT_OUTER_WEIGHT() {
        return 15;
    }
}

function dot(v, u) {
    let s = 0.0;
    for (let i = 0; i < v.length; i++) {
        s += v[i]*u[i];
    }
    return s;
}

let c = 1;
let d = 2;

class SVMModel {
    constructor(other) {
        this.W = [0, 1, -1];
        this.set(other);
    }

    set(other) {
        if (other != null) {
            this.W = other.W.map(w => w);
        }
    }

    score(x) {
        // let c = 1;
        // let d = 2;
        // let diff = [this.W[0] - x[0], this.W[1] - x[1], this.W[2] - x[2]];
        // return Math.exp(-Math.sqrt(dot(diff, diff)));
        // return Math.pow(dot(this.W, x) + c, d);
        // let r = Math.sqrt(x[1]*x[1] + x[2]*x[2]);
        // return dot(this.W, [-1, r, x]);
        return dot(this.W, x);
    }

    computeCost(xs, ys) {
        let hingeLoss = 0.0;
        let N = xs.length;

        // Compute Hinge Loss
        for (let i = 0; i < N; i++) {
            hingeLoss += Math.max(0, 1 - ys[i]*this.score(xs[i]));
        }
        hingeLoss = C * (hingeLoss / N);

        // Add to Cost SVM Margin Minimization
        let w = [this.W[1], this.W[2]];
        let marginLoss = (1/2) * dot(w, w);
        
        let totalLoss = hingeLoss + marginLoss;

        return {
            hingeLoss: hingeLoss,
            marginLoss: marginLoss,
            totalLoss: totalLoss
        };
    }

    computeSubGradient(xs, ys) {
        let W = this.W;
        let grads = [...this.W];
        let N = xs.length;

        for (let j = 0; j < W.length; j++) {
            // Compute SubGrad for Parameter j
            let grad = 0.0;
            for (let i = 0; i < N; i++) {
                if (Math.max(0, 1 - ys[i] * this.score(xs[i])) == 0) {
                    grad += W[j];
                }
                else {
                    grad += W[j] - C*ys[i]*xs[i][j];
                }
            }
            grad /= N;

            grads[j] = grad;
        }

        return grads;
    }

    toString() {
        return `${this.W[1].toFixed(4)}x + ${this.W[2].toFixed(4)}y - ${this.W[0].toFixed(4)}`;
    }

    getYFromX(x) {
        return (-this.W[1]*x + this.W[0]) / this.W[2];
    }

    get w() {
        return [this.W[1], this.W[2]];
    }
}

class SVMOptimizer {
    constructor(currentSVM, bestSVM, points, updateErrorGraphCb) {
        this.iter = 0;
        this.currentSVM = currentSVM;
        this.bestSVM = bestSVM;
        this.points = points;

        this.errorHistory = [];

        this.hingeLossHistory = [];
        this.marginLossHistory = [];

        this.bestLoss = 0;
        this.stochasticIndex = 0;

        this.updateErrorGraphCb = updateErrorGraphCb;
    }

    beforeTrainStart() {
        let initialLoss = this.currentSVM.computeCost(
            this.points.map(p => T.ext(p.x)),  // xs
            this.points.map(p => p.y)); // ys

        console.log("Initial Loss: " + initialLoss);
        this.bestLoss = initialLoss.totalLoss;
    }

    update() {
        shuffleArray(points);
        let ys = points.map(p => p.y);
        let xs = points.map(p => T.ext(p.x));
        

        if (stochasticLearning) {
            xs = xs.slice(stochasticIndex, stochasticIndex+1);
            ys = ys.slice(stochasticIndex, stochasticIndex+1);
        }
        
        let subgrad = this.currentSVM.computeSubGradient(xs, ys);
        for (let j = 0; j < subgrad.length; j++) {
            this.currentSVM.W[j] = this.currentSVM.W[j] - lr * subgrad[j];
        }

        if (stochasticLearning) {
            stochasticIndex = stochasticIndex + 1;
            if (stochasticIndex == points.length) {
                stochasticIndex = 0;
                this.iter += 1;
            }
        } else {
            this.iter += 1;
        }

        let loss = this.currentSVM.computeCost(xs, ys);
        let totalLoss = loss.totalLoss;
        this.errorHistory.push(loss.totalLoss);
        this.marginLossHistory.push(loss.marginLoss);
        this.hingeLossHistory.push(loss.hingeLoss);

        if (totalLoss < this.bestLoss) {
            this.bestLoss = totalLoss;
            this.bestSVM.set(this.currentSVM);
        }

        if (this.iter >= MAX_ITERS) {
            this.stopTraining();
        }
        let errN = this.errorHistory.length;
        if (errN > 2) {
            let currError = this.errorHistory[errN-1],
                prevError = this.errorHistory[errN-2];
            let relError = Math.abs(currError - prevError) / currError;
            if (relError < ERR_THRESHOLD) {
                this.stopTraining();
            }
        }
    }

    stopTraining() {
        training = false;
        console.log("Min Error: " + this.lastCost);
        console.log("Trained!");
        this.updateErrorGraphCb(this.errorHistory, this.hingeLossHistory, this.marginLossHistory);
    }

    get lastCost() {
        if (this.errorHistory.length > 0) {
            return this.errorHistory[this.errorHistory.length-1];
        }
        else {
            return null;
        }
    }
}