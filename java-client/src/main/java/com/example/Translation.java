package com.example;

import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.ParameterStore;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.reflect.TypeToken;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.Namespace;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Hello world!
 *
 */
public class Translation
{
    private static final Logger logger = LoggerFactory.getLogger(Translation.class);

    private final String source_word2index;
    private final String target_index2word;
    private final Map<String, Integer> params;

    private final String encoder;
    private final String decoder;

    public Translation(String source_word2index, String target_index2word, String param, String encoder, String decoder) throws Exception{
        this.source_word2index = source_word2index;
        this.target_index2word = target_index2word;
        DataLoader dl = new DataLoader();
        params = dl.load_param(param);
        this.encoder = encoder;
        this.decoder = decoder;
    }

    public String predict(String french) throws ModelException, TranslateException, IOException {
        Path path = Paths.get(source_word2index);
        Map<String, Long> wrd2idx;
        try (InputStream is = Files.newInputStream(path)) {
            String json = Utils.toString(is);
            Type mapType = new TypeToken<Map<String, Long>>() {}.getType();
            wrd2idx = JsonUtils.GSON.fromJson(json, mapType);
        }

        path = Paths.get(target_index2word);
        Map<String, String> idx2wrd;
        try (InputStream is = Files.newInputStream(path)) {
            String json = Utils.toString(is);
            Type mapType = new TypeToken<Map<String, String>>() {}.getType();
            idx2wrd = JsonUtils.GSON.fromJson(json, mapType);
        }

        Engine engine = Engine.getEngine("PyTorch");
        try (NDManager manager = engine.newBaseManager()) {
            try (ZooModel<NDList, NDList> encoder = getEncoderModel();
                 ZooModel<NDList, NDList> decoder = getDecoderModel()) {

                NDList toDecode = predictEncoder(french, encoder, wrd2idx, manager);
                String english = predictDecoder(toDecode, decoder, idx2wrd, manager);

                logger.info("French: {}", french);
                logger.info("English: {}", english);
                return english;
            }
        }
    }

    public ZooModel<NDList, NDList> getEncoderModel() throws ModelException, IOException {
        Path p1 = Paths.get("");
        Path p2 = p1.toAbsolutePath();
        String url = "file:///"+p2.toAbsolutePath();

        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optModelName(encoder)
                        .optEngine("PyTorch")
                        .build();
        return criteria.loadModel();
    }

    public ZooModel<NDList, NDList> getDecoderModel() throws ModelException, IOException {
        Path p1 = Paths.get("");
        Path p2 = p1.toAbsolutePath();
        String url = "file:///"+p2.toAbsolutePath();
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelUrls(url)
                        .optModelName(decoder)
                        .optEngine("PyTorch")
                        .build();
        return criteria.loadModel();
    }

    public NDList predictEncoder(
            String text,
            ZooModel<NDList, NDList> model,
            Map<String, Long> wrd2idx,
            NDManager manager) {
        // maps french input to id's from french file
        List<String> list = Collections.singletonList(text);
        PunctuationSeparator punc = new PunctuationSeparator();
        list = punc.preprocess(list);
        List<Long> inputs = new ArrayList<>();
        for (String word : list) {
            if (word.length() == 1 && !Character.isAlphabetic(word.charAt(0))) {
                continue;
            }
            Long id = wrd2idx.get(word.toLowerCase(Locale.FRENCH));
            if (id == null) {
                throw new IllegalArgumentException("Word \"" + word + "\" not found.");
            }
            inputs.add(id);
        }

        // for forwarding the model
        Shape inputShape = new Shape(1);
        Shape hiddenShape = new Shape(1, 1, 256);
        FloatBuffer fb = FloatBuffer.allocate(256);
        NDArray hiddenTensor = manager.create(fb, hiddenShape);
        long[] outputsShape = {params.get("MAX_LENGTH"), params.get("hidden_size")};
        FloatBuffer outputTensorBuffer = FloatBuffer.allocate(params.get("MAX_LENGTH") * params.get("hidden_size"));

        // for using the model
        Block block = model.getBlock();
        ParameterStore ps = new ParameterStore();

        // loops through forwarding of each word
        for (long input : inputs) {
            NDArray inputTensor = manager.create(new long[] {input}, inputShape);
            NDList inputTensorList = new NDList(inputTensor, hiddenTensor);
            NDList outputs = block.forward(ps, inputTensorList, false);
            NDArray outputTensor = outputs.get(0);
            outputTensorBuffer.put(outputTensor.toFloatArray());
            hiddenTensor = outputs.get(1);
        }
        outputTensorBuffer.rewind();
        NDArray outputsTensor = manager.create(outputTensorBuffer, new Shape(outputsShape));

        return new NDList(outputsTensor, hiddenTensor);
    }

    public String predictDecoder(
            NDList toDecode,
            ZooModel<NDList, NDList> model,
            Map<String, String> idx2wrd,
            NDManager manager) {
        // for forwarding the model
        Shape decoderInputShape = new Shape(1, 1);
        NDArray inputTensor = manager.create(new long[] {0}, decoderInputShape);
        ArrayList<Integer> result = new ArrayList<>(params.get("MAX_LENGTH"));
        NDArray outputsTensor = toDecode.get(0);
        NDArray hiddenTensor = toDecode.get(1);

        // for using the model
        Block block = model.getBlock();
        ParameterStore ps = new ParameterStore();

        // loops through forwarding of each word
        for (int i = 0; i < params.get("MAX_LENGTH"); i++) {
            NDList inputTensorList = new NDList(inputTensor, hiddenTensor, outputsTensor);
            NDList outputs = block.forward(ps, inputTensorList, false);
            NDArray outputTensor = outputs.get(0);
            hiddenTensor = outputs.get(1);
            float[] buf = outputTensor.toFloatArray();
            int topIdx = 0;
            double topVal = -Double.MAX_VALUE;
            for (int j = 0; j < buf.length; j++) {
                if (buf[j] > topVal) {
                    topVal = buf[j];
                    topIdx = j;
                }
            }

            if (topIdx == params.get("EOS_token")) {
                break;
            }

            result.add(topIdx);
            inputTensor = manager.create(new long[] {topIdx}, decoderInputShape);
        }

        StringBuilder sb = new StringBuilder();
        // map english words and create output string
        for (Integer word : result) {
            sb.append(idx2wrd.get(word.toString())).append(' ');
        }
        return sb.toString().trim();
    }
    public static void main( String[] args ) throws Exception
    {
        ArgumentParser parser = ArgumentParsers.newFor("seq2seq").build()
                .defaultHelp(true)
                .description("seq2seq client sample");

        parser.addArgument("--source_word2index").setDefault("model/fra_word2index.json").help("");
        parser.addArgument("--target_index2word").setDefault("model/eng_index2word.json").help("");
        parser.addArgument("--params").setDefault("model/params.json").help("");
        parser.addArgument("--encoder").setDefault("model/encoder_scripted.pt").help("");
        parser.addArgument("--decoder").setDefault("model/decoder_scripted.pt").help("");
        parser.addArgument("--input").setDefault("trop tard").help("");
        Namespace ns = parser.parseArgs(args);

        String source_word2index = ns.getString("source_word2index");
        String target_index2word = ns.getString("target_index2word");
        String params = ns.getString("params");
        String encoder = ns.getString("encoder");
        String decoder = ns.getString("decoder");
        String input = ns.getString("input");

        Translation translation = new Translation(source_word2index,target_index2word,params,encoder,decoder);
        String english = translation.predict(input);
        System.out.println(english);
    }
}
