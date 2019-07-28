package TwitterPreprocessing; /**
 * Created by iosifidis on 01.08.16.
 */

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

import java.io.IOException;

public final class GeneratePosNegDictionary {
    static Logger logger = Logger.getLogger(GeneratePosNegDictionary.class);

    public static class MyMapper extends Mapper<Object, Text, Text, IntWritable> {

        public void map(Object key, Text row, Context context) throws IOException, InterruptedException {

            final String items = row.toString();

            String after_preprocessed;

            String[] stringArray = items.split(",");

            try {
                String emoticon = stringArray[5];
                after_preprocessed = stringArray[7];


                if (emoticon.equals("positive") || emoticon.equals("negative")) {
//                    logger.info(after_preprocessed);
                    final String [] hashTags = stringArray[2].split("-");
                    for(String word : after_preprocessed.split(" ")){
                        // skip words is they are hash tags
                        boolean hashTag = false;
                        for(String tag : hashTags){
                            if(tag.equals(word)){
                                hashTag = true;
                                break;
                            }
                        }
                        if(hashTag){
                            continue;
                        }

                        if (emoticon.equals("positive")){
//                            logger.info(word);
                             context.write(new Text(word), new IntWritable(1));
                        }
                        else {
//                            logger.info(word);
                            context.write(new Text(word), new IntWritable(-1));
                        }
                    }

                }
            }catch (ArrayIndexOutOfBoundsException e){
                logger.error(e,e);
            }
        }
    }

    public static class MyReducer extends Reducer<Text,IntWritable,Text,Text> {

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException {
            int sumPos = 0;
            int sumNeg = 0;
            for (IntWritable val : values) {
                if (val.get() == 1){
                    sumPos += val.get();
                }
                else{
                    sumNeg += val.get();
                }
            }
            Text output = new Text(sumPos + "," + sumNeg);
            context.write(key, new Text(output));
        }
    }


    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();

        if (args.length != 2) {
            logger.error("NEED 2 DIRECTORY ARGUMENTS: -input1(posts/comments) -OUTPUT>");
            System.exit(1);
        }
        Job job = new Job(configuration, "GeneratePosNegDictionary");

        job.setJarByClass(GeneratePosNegDictionary.class);

        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));

        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
