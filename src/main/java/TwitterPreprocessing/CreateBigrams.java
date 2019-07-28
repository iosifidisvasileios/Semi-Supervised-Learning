package TwitterPreprocessing; /**
 * Created by iosifidis on 01.08.16.
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.IOException;

public final class CreateBigrams {

    static Logger logger = Logger.getLogger(CreateBigrams.class);

    private static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {

        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {
            final String items = row.toString();
            try {

                String[] stringArray = items.split(",");
                final String idKey = String.valueOf(Long.valueOf(stringArray[0].split("\t")[0]));

                String[] preprocessed = stringArray[7].split(" ");
                String bigrams = "";
                for (int index = 0; index < preprocessed.length - 1; index++){
                    bigrams += preprocessed[index] + "_" + preprocessed[index + 1] + " ";
                }
                bigrams = bigrams.substring(0, bigrams.length()-1);

                final String outpout = "," + stringArray[1] + "," + stringArray[2] + "," + stringArray[3] + ","
                        + stringArray[4] + "," + stringArray[5] + "," + stringArray[6] + "," + stringArray[7] + "," + stringArray[8] + "," + bigrams;

                context.write(new Text(idKey), new Text(outpout));

            }catch(NumberFormatException e){
                logger.error(e);
                logger.error(items);
            }catch (ArrayIndexOutOfBoundsException e){
                logger.error(e);
            }
        }
    }

    static class MyReducer extends Reducer<Text, Text, Text, Text> {
        protected void reduce(Text key, Text values, Context context) throws IOException, InterruptedException {
            context.write(key, new Text(values));
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration configuration = new Configuration();

        //        configuration.set("mapreduce.map.java.opts", "-Xmx16096m");

        if (args.length != 2) {
            logger.error("NEED 2 DIRECTORY ARGUMENTS: -input(tweets) -OUTPUT>");
            System.exit(1);
        }
        Job job = new Job(configuration, "CreateBigrams");

        job.setJarByClass(CreateBigrams.class);
        job.setMapperClass(MyMapper.class);
        job.setReducerClass(MyReducer.class);

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(Text.class);

        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
