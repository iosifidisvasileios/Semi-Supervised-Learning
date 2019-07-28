package TwitterPreprocessing; /**
 * Created by iosifidis on 01.08.16.
 */

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.log4j.Logger;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashSet;
import java.util.Set;

public final class RemoveLowFreq {

    static Logger logger = Logger.getLogger(RemoveLowFreq.class);

    private static class MyMapper extends Mapper<LongWritable, Text, Text, Text> {
        private Set<String> stopWords;

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {

            Configuration conf = context.getConfiguration();
            FileSystem fs = FileSystem.get(conf);

            Path path = new Path("hdfs://nameservice1/user/iosifidis/low_freq.txt");
            if (fs.exists(path)) {
                stopWords = new HashSet<>();
                BufferedReader br = null;

                FSDataInputStream fis = fs.open(path);
                br = new BufferedReader(new InputStreamReader(fis));
                String line = null;
                while ((line = br.readLine()) != null && line.trim().length() > 0) {
                    line = line.split("\t")[0];
                    stopWords.add(line);
                }
            }
        }

        protected void map(LongWritable key, Text row, Context context) throws IOException, InterruptedException {

            final String items = row.toString();
            try {

                String[] stringArray = items.split("\t");
                final String idKey = stringArray[0];

                String text = stringArray[3];
                String reconstruction = "";
                for (String word : text.split(" ")){
                    if (!stopWords.contains(word.toLowerCase())) {
                        reconstruction += word.toLowerCase() + " ";
                    }
                }

                if (reconstruction.split(" ").length >=4 ) {
                    String outpout =  stringArray[1] + "\t" + stringArray[2] + "\t" + reconstruction;
                    context.write(new Text(idKey), new Text(outpout));
                }

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

//        configuration.set("mapreduce.map.memory.mb", "63096");
//        configuration.set("mapreduce.map.java.opts", "-Xmx63096m");

        Job job = new Job(configuration, "Remove Low Frequency Words");

        job.setJarByClass(RemoveLowFreq.class);
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
