import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class HW3 {

    public static void main(String[] args) throws IOException {

        if (args.length != 3) {
            throw new IllegalArgumentException("USAGE: file_path num_diversity num_partitions");
        }

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // SPARK SETUP
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        SparkConf conf = new SparkConf(true).setAppName("Homework3");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("WARN");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // VARIABLES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        int K = Integer.parseInt(args[1]); //parameter for diversity maximization
        int L = Integer.parseInt(args[2]); // number of partitions
        String inputPath = args[0]; //file path

        //Start time
        long startTime = System.currentTimeMillis();
        //reading the points in a JavaRDD
        JavaRDD<Vector> inputPoints = sc.textFile(inputPath).map(G20HW3::strToVector).repartition(L).cache();
        long numdocs =  inputPoints.count(); //force the loading for avoiding lazy evaluation
        //Start time
        long endTime = System.currentTimeMillis();

        System.out.println("Number of points = " + numdocs);
        System.out.println("K = " + K);
        System.out.println("L = " + L);
        System.out.println("Initialization time = " + (endTime-startTime)+"ms\n");


        ArrayList<Vector> fin = runMapReduce(inputPoints, K, L);
        double averageDistance = measure(fin);
        System.out.println("Average distance = " + averageDistance);

    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // METHOD runSequential
    // Sequential 2-approximation based on matching
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    public static ArrayList<Vector> runSequential(final ArrayList<Vector> points, int k) {

        final int n = points.size();
        if (k >= n) {
            return points;
        }

        ArrayList<Vector> result = new ArrayList<>(k);
        boolean[] candidates = new boolean[n];
        Arrays.fill(candidates, true);
        for (int iter=0; iter<k/2; iter++) {
            // Find the maximum distance pair among the candidates
            double maxDist = 0;
            int maxI = 0;
            int maxJ = 0;
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    for (int j = i+1; j < n; j++) {
                        if (candidates[j]) {
                            // Use squared euclidean distance to avoid an sqrt computation!
                            double d = Vectors.sqdist(points.get(i), points.get(j));
                            if (d > maxDist) {
                                maxDist = d;
                                maxI = i;
                                maxJ = j;
                            }
                        }
                    }
                }
            }
            // Add the points maximizing the distance to the solution
            result.add(points.get(maxI));
            result.add(points.get(maxJ));
            // Remove them from the set of candidates
            candidates[maxI] = false;
            candidates[maxJ] = false;
        }
        // Add an arbitrary point to the solution, if k is odd.
        if (k % 2 != 0) {
            for (int i = 0; i < n; i++) {
                if (candidates[i]) {
                    result.add(points.get(i));
                    break;
                }
            }
        }
        if (result.size() != k) {
            throw new IllegalStateException("Result of the wrong size");
        }
        return result;

    } // END runSequential

    /*implements the 4-approximation MapReduce algorithm for diversity maximization
    * Round 1: subdivides pointsRDD into L partitions and extracts k points from each partition using
    * the Farthest-First Traversal algorithm.
    * Hints :  • Recycle the implementation of FFT algorithm developed for HW 2;
    * • For the partitioning, invoking repartition(L) when the RDD was created, we can use the Spark Partitions,
    * accessing them through the mapPartition method.
    * Round 2: collects the L*k points extracted in Round 1 from the partitions into a set called coreset and returns
    * the k points computed by runSequential(coreset,k)
    * */

    public static ArrayList<Vector> runMapReduce(JavaRDD<Vector> pointsRDD,int k, int L){
        long startTime = System.currentTimeMillis();

        JavaRDD<Vector> cent = pointsRDD.mapPartitions((vectorIterator)->{
            ArrayList<Vector> temp = new ArrayList<>();
            while (vectorIterator.hasNext()){
                temp.add(vectorIterator.next());
            }
            return kCenterMPD(temp,k).iterator();
        });
        long endTime = System.currentTimeMillis();
        System.out.println("Runtime of Round 1 = " + (endTime-startTime)+"ms");

        startTime = System.currentTimeMillis();
        List<Vector> out = cent.collect();
        endTime = System.currentTimeMillis();
        long par1 = endTime-startTime; //first partial interval
        ArrayList<Vector> coreset = new ArrayList<>();
        for (int i=0; i<out.size();i++){
            coreset.add(out.get(i));
        }
        startTime = System.currentTimeMillis();
        ArrayList<Vector> selectedPoints = runSequential(coreset, k);
        endTime = System.currentTimeMillis();
        long par2 = endTime-startTime; //second partial interval ---> for avoiding to count the time for the FOR loop
        System.out.println("Runtime of Round 2 = " + (par1+par2) +"ms");

        return selectedPoints;
    }

    //receives in input a set of points (pointSet) and computes the average distance between all pairs of points.
    public static Double measure(ArrayList<Vector> pointsSet){
        Double s = 0.0;
        int d = 0;
        for(int i = 0;i<pointsSet.size();i++){
            for (int j = i+1; j<pointsSet.size();j++){
                s = s + Math.sqrt(Vectors.sqdist(pointsSet.get(i),pointsSet.get(j)));
                d += 1;
            }
        }
        return s/d;
    }

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) throws IOException{
        if(k>=S.size()){
            throw new IllegalArgumentException("Integer k greater than the cardinality of input set");
        }
        Random rand = new Random();
        rand.setSeed(1238164);
        //array of the distances of each point to the closest center
        ArrayList<Double> minDists = new ArrayList<>(S.size());
        ArrayList<Vector> centers = new ArrayList<>();
        //put random point in centers
        centers.add( S.get(rand.nextInt(S.size())) );
        double maxDist, dist;
        int idx=0;
        for(int j=0; j<k-1; j++) {
            // max distance from the current set of centers
            maxDist=0;
            for(int i=0; i<S.size(); i++) {
                //distance between
                dist=Vectors.sqdist(centers.get(j), S.get(i));
                //only for the first center fill minDists with the distances between it and every other point
                if(j==0) {
                    minDists.add(i, dist);
                    //for the other centers update minDists only if a smaller distance is found
                }else{
                    minDists.set( i, Math.min(minDists.get(i), dist) );
                }
                //if a bigger distance is found update the next center to the current point
                if (minDists.get(i) > maxDist) {
                    maxDist = minDists.get(i);
                    idx=i;
                }
            }
            centers.add(S.get(idx));
        }
        return centers;
    }
    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }
}
