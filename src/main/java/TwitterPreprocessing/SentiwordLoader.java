package TwitterPreprocessing;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;

/**
 * Created by iosifidis on 13.08.16.
 */
public class SentiwordLoader {
    static Logger logger = Logger.getLogger(SentiwordLoader.class);
    private final HashMap<String, Double> dictionary = new HashMap<>();

    public SentiwordLoader() {

        HashMap<String, HashMap<Integer, Double>> tmpDictionary = new HashMap<>();
        FileInputStream fstream = null;
        try {

            Path pt=new Path("hdfs://nameservice1/user/iosifidis/SentiWordnet.txt");
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
//            double sum = 0.0;
            for(int key_2 : temp.keySet()){
                score += temp.get(key_2)/ Double.valueOf(key_2);
//                sum += 1.0 / Double.valueOf(key_2);
            }
//            score /= sum;
            dictionary.put(key, score);
        }
    }

    public HashMap<String, Double> getDictionary() {
        return dictionary;
    }
}

