import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.*;

import static java.lang.Long.parseLong;

public class HW1 {

    static long maximumSize = 0;
    public static void main(String[] args) throws IOException {

        System.setProperty("hadoop.home.dir", "C:\\UNIPD\\big_data");

        if (args.length != 2) {
            throw new IllegalArgumentException("USAGE: num_partitions file_path");
        }

        SparkConf conf = new SparkConf(true).setAppName("Homework1");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // Read number of partitions
        int K = Integer.parseInt(args[0]);

        // Read input file and subdivide it into K random partitions
        JavaRDD<String> docs = sc.textFile(args[1]).repartition(K);

        long numdocs, numwords;
        numdocs = docs.count();
        System.out.println("Number of documents = " + numdocs);
        JavaPairRDD<String, Long> count;
        ArrayList<Long> ml = new ArrayList<>();  //array delle partition sizes


        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CLASS COUNT
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        Random randomGenerator = new Random();
        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] entry = document.split(" ");
                    HashMap<Long, String> counts = new HashMap<>();
                    ArrayList<Tuple2<Long, Tuple2<Long, String>>> pairs = new ArrayList<>();

                    /*for(String line : entry){
                        String[] lineSplitted = line.split(" ");

                    }*/
                    counts.put(parseLong(entry[0]), entry[1]);
                    for (Map.Entry<Long, String> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>((e.getKey()%K), new Tuple2<>(e.getKey(), e.getValue())));  //add deterministic id number to the tuples
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R1)
                .flatMapToPair((triplet) -> {
                    HashMap<String, Long> counts = new HashMap<>();
                    for (Tuple2<Long, String> c : triplet._2()) {
                        counts.put(c._2(), 1L + counts.getOrDefault(c._2(), 0L));
                    }
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()    // <-- REDUCE PHASE (R2)
                .sortByKey()
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });

        System.out.println("output 1: " + count.collect().toString());

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // CLASS COUNT with mapPartitions
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        count = docs
                .flatMapToPair((document) -> {    // <-- MAP PHASE (R1)
                    String[] entry = document.split(" ");
                    HashMap<Long, String> counts = new HashMap<>();
                    ArrayList<Tuple2<Long, String>> pairs = new ArrayList<>();
                    counts.put(parseLong(entry[0]), entry[1]);
                    for (Map.Entry<Long, String> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .mapPartitionsToPair((wc) -> {    // <-- REDUCE PHASE (R1)
                    HashMap<String, Long> counts = new HashMap<>();
                    Long counter = 0L;
                    while (wc.hasNext()){
                        Tuple2<Long, String> tuple = wc.next();
                        counts.put(tuple._2(), 1L + counts.getOrDefault(tuple._2(), 0L));
                        counter++;
                    }
                    System.out.println("partition size: " + counter);
                    if(counter > maximumSize){
                        maximumSize = counter;
                    }
                    
                    ArrayList<Tuple2<String, Long>> pairs = new ArrayList<>();
                    for (Map.Entry<String, Long> e : counts.entrySet()) {
                        pairs.add(new Tuple2<>(e.getKey(), e.getValue()));
                    }
                    return pairs.iterator();
                })
                .groupByKey()     // <-- REDUCE PHASE (R2)
                .mapValues((it) -> {
                    long sum = 0;
                    for (long c : it) {
                        sum += c;
                    }
                    return sum;
                });
        Tuple2<String, Long> maximum = new Tuple2<>("FirstElement", 0L);
        for(Tuple2<String, Long> e : count.collect()){
            if(e._2>maximum._2){
                maximum = e;
            }
        }
        System.out.println("output 2: " + maximum.toString());
        System.out.println(maximumSize);
        
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // COMPUTE MAX FREQUENT CLASS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        JavaPairRDD<String, Long> temp = count;
        Long max_freq = temp
                .map((tuple) -> tuple._2())
                .reduce((x, y) -> Math.max(x,y));

        //System.out.println("Maximum freq:" + max_freq);
        System.out.println("Most frequent class: "+ count.filter((tuple)-> tuple._2().equals(max_freq)).collect());
    }
}
