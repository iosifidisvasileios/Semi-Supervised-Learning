package TwitterPreprocessing;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.HashSet;

/**
 * Created by iosifidis on 13.08.16.
 */
public class HashtagSentiment {
    static Logger logger = Logger.getLogger(HashtagSentiment.class);
    private final HashMap<String, Double> dictionary = new HashMap<>();
    private final HashSet<String> positive = new HashSet<>();
    private final HashSet<String> negative = new HashSet<>();

    public HashtagSentiment() {


        try {

            Path pt=new Path("hdfs://nameservice1/user/iosifidis/HashtagSentiment.txt");
            FileSystem fs = FileSystem.get(new Configuration());
            BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(pt)));

//            FileInputStream fstream = new FileInputStream("HashtagSentiment.txt");
//            BufferedReader br=new BufferedReader(new InputStreamReader(fstream));

            String strLine;

            while ((strLine = br.readLine()) != null) {
                if(!strLine.trim().startsWith("#")){continue;}
                String[] array = strLine.split("\t");
                String hashtag = array[0];
                double value = Double.valueOf(array[1]);
                if (value < 0){
                    negative.add(hashtag);
                }else if (value > 0){
                    positive.add(hashtag);
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public HashSet<String> getPositiveSet() {return positive; }
    public HashSet<String> getNegativeSet() {return negative; }

}

