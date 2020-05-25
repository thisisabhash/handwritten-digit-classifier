import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient
    private double targetValue = 0.0;
    private double sumOfPartialDelta = 0.0; // for hidden layer

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {   //Not an input or bias node

            double in = 0.0;
            // summation of w_i * a_i
            for (int i = 0; i < parents.size(); i++) {
                NodeWeightPair nodeWeightPair = parents.get(i);
                in = in + (nodeWeightPair.weight * nodeWeightPair.node.getOutput());
            }

            if (type == 2) {
                //hidden node
                //use ReLU
                outputValue = Math.max(0.0, in);
            } else {
                // output node
                // use Softmax
                outputValue = Math.exp(in);
            }
        }
    }

    // method to update output value of output nodes
    public void updateOutputValue(double sumOfExponents) {
        if (type == 4) {
            outputValue /= sumOfExponents;
        }
    }

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    //Calculate the delta value of a node.
    public void calculateDelta() {
        if (type == 2 || type == 4) {
            if (type == 4) {
                delta = targetValue - outputValue;
                for (int i = 0; i < parents.size(); i++) {
                    NodeWeightPair nodeWeightPair = parents.get(i);
                    Node hiddenNode = nodeWeightPair.node;
                    hiddenNode.updateSumOfPartialDelta(delta * nodeWeightPair.weight);
                }
            } else {
                // type 2 hidden layer
                // ReLU activation
                // derivative is 0 if output value is <=0, so delta becomes 0
                // else derivative is 1, delta becomes sumOfPartialDelta
                if (outputValue <= 0.0) {
                    delta = 0;
                } else {
                    delta = sumOfPartialDelta;
                }
            }
        }
    }

    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
            for (int i = 0; i < parents.size(); i++) {
                NodeWeightPair nodeWeightPair = parents.get(i);
                Node node = nodeWeightPair.node;
                nodeWeightPair.weight += learningRate * node.getOutput() * delta;
            }
        }
    }

    public void setTargetValue(double targetValue) {
        if (type == 4) {
            this.targetValue = targetValue;
        }
    }

    // only for hidden layer
    public void updateSumOfPartialDelta(double partialDelta) {
        if (type == 2) {
            sumOfPartialDelta += partialDelta;
        }
    }

    // after each training instance is done, call this to reset value of sumOfPartialDelta
    public void resetSumOfPartialDelta() {
        if (type == 2) {
            sumOfPartialDelta = 0.0;
        }
    }
}


