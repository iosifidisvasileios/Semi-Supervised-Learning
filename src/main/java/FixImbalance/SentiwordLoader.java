package FixImbalance;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by iosifidis on 13.08.16.
 */
public class SentiwordLoader {

    private final HashSet<String> dictionary = new HashSet<String>();

    public SentiwordLoader() {

        HashMap<String, HashMap<Integer, Double>> tmpDictionary = new HashMap<String, HashMap<Integer, Double>>();
        FileInputStream fstream = null;
        try {

            Path pt=new Path("SentiWordnet.txt");
//            Path pt=new Path("hdfs://nameservice1/user/iosifidis/SentiWordnet.txt");
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));

            String strLine;

            while ((strLine = br.readLine()) != null) {
                if(strLine.trim().startsWith("#")){continue;}
                String[] array = strLine.split("\t");
                String wordTypeMarker = array[0];
                Double synsetScore = Double.parseDouble(array[2]) - Double.parseDouble(array[3]);
                String[] synTermsSplit = array[4].split(" ");

                for (String term : synTermsSplit) {
                    String[] synTermAndRank = term.split("#");
                    String synTerm = synTermAndRank[0] + "#" + wordTypeMarker;
                    int synTermRank = Integer.parseInt(synTermAndRank[1]);

                    if (!tmpDictionary.containsKey(synTerm)) {
                        tmpDictionary.put(synTerm, new HashMap<Integer, Double>());
                    }
                    tmpDictionary.get(synTerm).put(synTermRank, synsetScore);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        for(String key : tmpDictionary.keySet()){
            HashMap<Integer,Double> temp = tmpDictionary.get(key);

            double score = 0.0;
            for(int key_2 : temp.keySet()){
                score += temp.get(key_2)/ (double) key_2;
            }
            if (score != 0)
                dictionary.add(key.split("#")[0]);
        }
    }

    public HashSet<String> getDictionary() {
        System.out.println("Size of dictionary is !!! ~>>>>>> : " + dictionary.size());

//        dictionary.forEach((s) -> {
//            System.out.println(s);
//        });
        return dictionary;
    }
}

