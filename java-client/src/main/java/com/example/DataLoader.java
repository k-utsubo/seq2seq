package com.example;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

public class DataLoader {


    // resouceを読むのは時間がかかるのでキャッシュしておく
    private JsonNode getNode(Class cls, String resource) throws Exception{
        ObjectMapper mapper = new ObjectMapper();
        JsonNode node = mapper.readTree(cls.getResourceAsStream(resource));
        return node;
    }

    private JsonNode getNode(String resource) throws Exception{
        ObjectMapper mapper = new ObjectMapper();
        JsonNode node = mapper.readTree(new File(resource));
        return node;
    }

    private List<String> keys(JsonNode node){
        //https://www.baeldung.com/java-jsonnode-get-keys
        List<String> keys = new ArrayList<>();
        Iterator<String> iterator = node.fieldNames();
        iterator.forEachRemaining(e -> keys.add(e));
        return keys;
    }


    Map<String, Integer> load_param(String resource) throws Exception {
        JsonNode node = getNode(resource);
        Map<String, Integer> dict =new HashMap<>();
        List<String> keys = keys(node);
        for(String key:keys){  // key は漢字
            JsonNode valNode = node.get(key);
            Integer ival = valNode.asInt();
            dict.put(key, ival);
        }
        return dict;
    }

}
