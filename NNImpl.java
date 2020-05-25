import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
        // set the input value in the input nodes from the training instance
        for (int j = 0; j < instance.attributes.size(); j++) {
            inputNodes.get(j).setInput(instance.attributes.get(j));
        }

        //set the target value in output nodes
        for (int j = 0; j < instance.classValues.size(); j++) {
            outputNodes.get(j).setTargetValue((double) instance.classValues.get(j));
        }

        // calculate values for hidden nodes
        for (int j = 0; j < hiddenNodes.size(); j++) {
            // for each hidden node
            Node hiddenNode = hiddenNodes.get(j);
            hiddenNode.calculateOutput();
        }

        double sumOfExponents = 0.0;
        //calculate values for output nodes
        for (int j = 0; j < outputNodes.size(); j++) {
            // for each output node
            Node outputNode = outputNodes.get(j);
            outputNode.calculateOutput();
            sumOfExponents += outputNode.getOutput();
        }

        //update output values of output nodes
        for (int j = 0; j < outputNodes.size(); j++) {
            Node outputNode = outputNodes.get(j);
            outputNode.updateOutputValue(sumOfExponents);
        }

        int index = 0;
        double maxValue = outputNodes.get(0).getOutput();
        for (int j = 1; j < outputNodes.size(); j++) {
            if (outputNodes.get(j).getOutput() > maxValue) {
                maxValue = outputNodes.get(j).getOutput();
                index = j;
            }
        }

        return index;
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
        // For each epoch, call setInputValue on input nodes
        for (int i = 0; i < maxEpoch; i++) {
            Collections.shuffle(trainingSet, random);

            // get each training instance
            for (int k = 0; k < trainingSet.size(); k++) {

                Instance instance = trainingSet.get(k);

                // set the input value in the input nodes from the training instance
                for (int j = 0; j < instance.attributes.size(); j++) {
                    inputNodes.get(j).setInput(instance.attributes.get(j));
                }

                //set the target value in output nodes
                for (int j = 0; j < instance.classValues.size(); j++) {
                    outputNodes.get(j).setTargetValue((double) instance.classValues.get(j));
                }

                // calculate values for hidden nodes
                for (int j = 0; j < hiddenNodes.size(); j++) {
                    // for each hidden node
                    Node hiddenNode = hiddenNodes.get(j);
                    hiddenNode.calculateOutput();
                }

                //calculate values for output nodes
                double sumOfExponents = 0.0;
                for (int j = 0; j < outputNodes.size(); j++) {
                    // for each output node
                    Node outputNode = outputNodes.get(j);
                    outputNode.calculateOutput();
                    sumOfExponents += outputNode.getOutput();
                }

                //update output values of output nodes
                for (int j = 0; j < outputNodes.size(); j++) {
                    Node outputNode = outputNodes.get(j);
                    outputNode.updateOutputValue(sumOfExponents);
                }

                // calculate delta values for output nodes
                for (int j = 0; j < outputNodes.size(); j++) {
                    Node outputNode = outputNodes.get(j);
                    outputNode.calculateDelta();
                }

                // calculate delta values for hidden nodes
                for (int j = 0; j < hiddenNodes.size(); j++) {
                    Node hiddenNode = hiddenNodes.get(j);
                    hiddenNode.calculateDelta();
                    hiddenNode.resetSumOfPartialDelta();
                }

                // update weights going from input layer to hidden layer
                for (int j = 0; j < hiddenNodes.size(); j++) {
                    Node hiddenNode = hiddenNodes.get(j);
                    hiddenNode.updateWeight(learningRate);
                }

                // update weights going from hidden layer to output layer
                for (int j = 0; j < outputNodes.size(); j++) {
                    Node outputNode = outputNodes.get(j);
                    outputNode.updateWeight(learningRate);
                }

                /*if (k == 0 && i==0) {
                    for (int j = 0; j < outputNodes.size(); j++) {
                        Node outputNode = outputNodes.get(j);
                        for (NodeWeightPair pair : outputNode.parents) {
                            System.out.println(pair.weight);
                        }
                    }
                }

                if (k == 0 && i == 0) {
                    for (int j = 0; j < hiddenNodes.size(); j++) {
                        Node hiddenNode = hiddenNodes.get(j);
                        if (hiddenNode.parents != null) {
                            for (NodeWeightPair pair : hiddenNode.parents) {
                                System.out.println(pair.weight);
                            }
                        }
                    }
                }*/
            }

           /* if (i==29) {
                for (int j = 0; j < outputNodes.size(); j++) {
                    Node outputNode = outputNodes.get(j);
                    for (NodeWeightPair pair : outputNode.parents) {
                        System.out.println(pair.weight);
                    }
                }
            }*/

            double totalLoss = 0.0;
            // Calculate loss and sum for each training instance, and then take average
            for (int k = 0; k < trainingSet.size(); k++) {
                Instance instance = trainingSet.get(k);
                totalLoss += loss(instance);
            }
            totalLoss /= trainingSet.size();
            System.out.println("Epoch: " + i + ", " + "Loss: " + String.format("%.3e", totalLoss));
        }
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
        // set the input value in the input nodes from the training instance
        for (int j = 0; j < instance.attributes.size(); j++) {
            inputNodes.get(j).setInput(instance.attributes.get(j));
        }

        //set the target value in output nodes
        for (int j = 0; j < instance.classValues.size(); j++) {
            outputNodes.get(j).setTargetValue((double) instance.classValues.get(j));
        }

        // calculate values for hidden nodes
        for (int j = 0; j < hiddenNodes.size(); j++) {
            // for each hidden node
            Node hiddenNode = hiddenNodes.get(j);
            hiddenNode.calculateOutput();
        }

        double sumOfExponents = 0.0;
        //calculate values for output nodes
        for (int j = 0; j < outputNodes.size(); j++) {
            // for each output node
            Node outputNode = outputNodes.get(j);
            outputNode.calculateOutput();
            sumOfExponents += outputNode.getOutput();
        }

        //update output values of output nodes
        for (int j = 0; j < outputNodes.size(); j++) {
            Node outputNode = outputNodes.get(j);
            outputNode.updateOutputValue(sumOfExponents);
        }

        double lossPerInstance = 0.0;
        for (int j = 0; j < instance.classValues.size(); j++) {
            double y = (double) instance.classValues.get(j);
            Node outputNode = outputNodes.get(j);
            lossPerInstance = lossPerInstance + (-1) * y * Math.log(outputNode.getOutput());
        }

        return lossPerInstance;
    }
}
