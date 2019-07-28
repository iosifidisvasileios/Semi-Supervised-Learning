package TwitterPreprocessing; /**
 * Created by iosifidis on 01.08.16.
 */

import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;
import org.influxdb.InfluxDB;
import org.influxdb.dto.Point;

import java.io.IOException;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.concurrent.TimeUnit;

public final class WordVariationBasedOnSentiwordNet {
    static Logger logger = Logger.getLogger(WordVariationBasedOnSentiwordNet.class);
    public static final HashSet<String> famous = new FamousWords().getFamousArray();
    public static final InfluxDB connector = new InfluxConnector().getInfluxDB();
    public static final MaxentTagger tagger = new TaggerLoader().getTagger();
    public static final HashMap<String,Double> sentiDictionary = new SentiwordLoader().getDictionary();
    public static final String dbName = "timeseries1";
    public static class MyMapper extends Mapper<Object, Text, Text, IntWritable> {

        public void map(Object key, Text row, Context context) throws IOException, InterruptedException {
            final String items = row.toString();
            try {
                String[] stringArray = items.split(",");
                String after_preprocessed = stringArray[7];
                for (String word : famous){

                    if (after_preprocessed.contains(word)){
                        final Date date = new Date(stringArray[1]);
                        String emoticon = stringArray[5];
                        int sentiment = getSentimentBasedOnSentWordNet(emoticon, after_preprocessed);

                        Point point = Point.measurement(word)
                                .time(date.getTime(), TimeUnit.MILLISECONDS)
                                .addField("sentiment", sentiment)
                                .build();

                        if(sentiment == 1 && emoticon.equals("positive")) {
                            connector.write(dbName, "default", point);
                        }else if(sentiment == -1 && emoticon.equals("negative")){
                            connector.write(dbName, "default", point);
                        }

                    }
                }
            }catch (ArrayIndexOutOfBoundsException e){
                logger.error(e,e);
            }
        }

        private int getSentimentBasedOnSentWordNet(String emoticon, String after_preprocessed) {

            if (emoticon.equals("positive") || emoticon.equals("negative")) {
                String tmp = "";
                int counter = 0;
                double score = 0.0;

                for (String word : after_preprocessed.split(" ")) {
                    boolean flag = false;
                    if (word.contains("_")) {
                        flag = true;
                        word = word.split("_")[1];
                    }
                    String temporal = tagger.tagString(word);
                    if (flag) {
                        temporal = "NOT_" + temporal;
                    }
                    tmp += temporal;
//                        tmp += tagger.tagString(word);
                }

                for (String word : tmp.trim().split(" ")) {
                    boolean flag_2 = false;
                    if (word.split("_").length == 3) {
//                        logger.info(word);
                        flag_2 = true;
                        word = word.replace("NOT_", "");
                    }

                    String term = word.split("_")[1];
                    String typeOfSpeech;

                    switch (term) {
                        case "JJ":
                        case "JJR":
                        case "JJS":
                            typeOfSpeech = "a";
                            break;
                        case "NN":
                        case "NNS":
                        case "NNP":
                        case "NNPS":
                            typeOfSpeech = "n";
                            break;
                        case "RB":
                        case "RBR":
                        case "RBS":
                            typeOfSpeech = "r";
                            break;
                        case "VB":
                        case "VBD":
                        case "VBG":
                        case "VBN":
                        case "VBP":
                        case "VBZ":
                            typeOfSpeech = "v";
                            break;
                        default:
                            typeOfSpeech = "null";
                    }

                    if (!typeOfSpeech.equals("null")) {
                        if (sentiDictionary.containsKey(word.split("_")[0] + "#" + typeOfSpeech)) {
                            if (flag_2) {
                                score -= sentiDictionary.get(word.split("_")[0] + "#" + typeOfSpeech);
                            } else {
                                score += sentiDictionary.get(word.split("_")[0] + "#" + typeOfSpeech);
                            }
                            counter++;
                        }
                    }
                }
                if (score != 0.0) {
                    score /= Double.valueOf(counter);
                }
                if (score < 0 ){
                    return -1;
                }else if(score > 0 ){
                    return 1;
                }else{
                    return 0;
                }
            }
            return 0;
        }
    }

    public static class MyReducer extends Reducer<Text,IntWritable,Text,IntWritable> {
        private IntWritable result = new IntWritable();
        public void reduce(Text key, Iterable<IntWritable> values, Context context
        ) throws IOException, InterruptedException {
            context.write(key, result);
        }
    }
    
    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();

        if (args.length != 2) {
            logger.error("NEED 2 DIRECTORY ARGUMENTS: -input1(posts/comments) -OUTPUT>");
            System.exit(1);
        }
        Job job = new Job(configuration, "WordVariationBasedOnSentiwordNet");

        job.setJarByClass(WordVariationBasedOnSentiwordNet.class);

        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        job.setNumReduceTasks(0);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
