package ai.certifai.Assignment.FishMarket;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class FishMarket {

    private static final int totalData = 160;
    private static final double ratioTrainTestSplit = 0.8;

    // Training info
    private static final int epoch = 1000;

    public static void main(String[] args) throws IOException, InterruptedException {

        RecordReader rr = loadData();

        List<List<Writable>> rawTrainData = new ArrayList<>();
        List<List<Writable>> rawTestData = new ArrayList<>();


        int numTrainData = (int) Math.round(ratioTrainTestSplit * totalData);
        int idx = 0;
        while (rr.hasNext()) {
            if(idx < numTrainData) {
                rawTrainData.add(rr.next());
            } else {
                rawTestData.add(rr.next());
            }
            idx++;
        }

        System.out.println("Total train Data " + rawTrainData.size());
        System.out.println("Total test Data " + rawTestData.size());

        List<List<Writable>> transformedTrainData = transformData(rawTrainData);
        List<List<Writable>> transformedTestData = transformData(rawTestData);

        DataSetIterator trainData = makeIterator(transformedTrainData);
        DataSetIterator testData = makeIterator(transformedTestData);

        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.setPreProcessor(normalizer);
        testData.setPreProcessor(normalizer);


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .updater(new Nesterovs(0.001, Nesterovs.DEFAULT_NESTEROV_MOMENTUM))
                .weightInit(WeightInit.XAVIER)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nIn(6)
                        .nOut(32)
                        .build())
                .layer(1, new DropoutLayer(0.3))
                .layer(2, new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(64)
                        .build())
                .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .nOut(7)
                        .activation(Activation.SOFTMAX)
                        .build())
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        StatsStorage storage = new InMemoryStatsStorage();
        UIServer server = UIServer.getInstance();
        server.attach(storage);

        // Set model listeners
        model.setListeners(new StatsListener(storage, 10));


        Evaluation eval;
        for(int i=0; i < epoch; i++) {
            model.fit(trainData);
            eval = model.evaluate(testData);
            System.out.println("EPOCH: " + i + " Accuracy: " + eval.accuracy());
            testData.reset();
            trainData.reset();
        }

        System.out.println("=== Train data evaluation ===");
        eval = model.evaluate(trainData);
        System.out.println(eval.stats());

        System.out.println("=== Test data evaluation ===");
        eval = model.evaluate(testData);
        System.out.println(eval.stats());

    }

    private static RecordReader loadData() throws IOException, InterruptedException {

        int numLinesToSkip = 1;
        char delimiter = ',';

        File inputFile = new ClassPathResource("Fish/Fish.csv").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);


        RecordReader rr = new CSVRecordReader(numLinesToSkip, delimiter);
        rr.initialize(fileSplit);

        return rr;
    }

    private static List<List<Writable>> transformData(List<List<Writable>> data) {

        Schema inputDataSchema = new Schema.Builder()
                .addColumnCategorical("Species", Arrays.asList("Bream", "Parkki", "Perch", "Pike", "Roach", "Smelt", "Whitefish"))
                .addColumnsFloat("Weight", "Length1", "Length2", "Length3", "Height", "Width")
                .build();

        System.out.println(inputDataSchema);


        TransformProcess tp = new TransformProcess.Builder(inputDataSchema)
                .categoricalToInteger("Species")
                .build();

        return LocalTransformExecutor.execute(data, tp);
    }

    private static DataSetIterator makeIterator(List<List<Writable>> data) {

        int labelIndex = 0;
        int numClasses = 7;

        RecordReader collectionRecordReaderData = new CollectionRecordReader(data);

        return new RecordReaderDataSetIterator(collectionRecordReaderData, data.size(), labelIndex, numClasses);
    }
}