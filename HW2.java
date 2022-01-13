import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.Vector;

import java.io.IOException;
import java.util.*;

import java.nio.file.Files;
import java.nio.file.Paths;

public class G20HW2 {

    static final long SEED = 1238164;

    public static void main(String[] args) throws IOException {

        String filename = args[0];
        ArrayList<Vector> inputPoints;

        //load file
        inputPoints = readVectorsSeq(filename);

        int k = Integer.parseInt(args[1]);

        //run methods and print times
        System.out.println("EXACT ALGORITHM");
        double startTime = System.nanoTime();
        System.out.println("Max distance = "+ exactMPD(inputPoints));  //TAKES A LOT OF TIME FOR MEDIUM AND LARGE
        System.out.println("Running time = " + (System.nanoTime()-startTime)/1000000+" ms\n");

        System.out.println("2-APPROXIMATION ALGORITHM");
        System.out.println("K = "+k);
        startTime = System.nanoTime();
        System.out.println("Max distance = "+twoApproxMPD(inputPoints,k));
        System.out.println("Running time = "+ (System.nanoTime()-startTime)/1000000+" ms\n");

        System.out.println("k-CENTER-BASED ALGORITHM");
        System.out.println("K = "+k);
        startTime = System.nanoTime();
        ArrayList<Vector> centers = kCenterMPD(inputPoints, k);
        //System.out.println("CENTERS: "+centers);
        System.out.println("Max distance = "+exactMPD(centers));
        System.out.println("Running time = " + (System.nanoTime()-startTime)/1000000+" ms\n");
    }

    public static double exactMPD(ArrayList<Vector> S){
        double maxDist=0;
        //try every possible pair and keep maximum distance found
        for(Vector vector1 : S){
            for (Vector vector2 : S) {
                double dist = Math.sqrt(Vectors.sqdist(vector1, vector2));
                if (dist > maxDist) {

                    maxDist = dist;
                }
            }
        }
        return  maxDist;
    }

    public static double twoApproxMPD(ArrayList<Vector> S, int k) throws IOException{
        if(k>=S.size()){
            throw new IllegalArgumentException("Integer k greater than the cardinality of input set");
        }
        //create copy to avoid modifications to the original
        ArrayList<Vector> copy = new ArrayList<>(S);
        Random rand = new Random();
        rand.setSeed(SEED);
        double dist, maxDist=0;
        //move k vectors to subset
        ArrayList<Vector> subset = new ArrayList<>();
        for(int i=0;i<k;i++){
            subset.add(copy.remove(rand.nextInt(copy.size())));
        }
        //check distance between all pairs {(v1,v2): v1 € S, v2 € subset}
        for (Vector v1 : copy) {
            for (Vector v2 : subset) {
                dist = Math.sqrt(Vectors.sqdist(v1, v2));
                if (dist > maxDist) {
                    maxDist = dist;
                }
            }
        }
        return maxDist;
    }

    public static ArrayList<Vector> kCenterMPD(ArrayList<Vector> S, int k) throws IOException{
        if(k>=S.size()){
            throw new IllegalArgumentException("Integer k greater than the cardinality of input set");
        }
        Random rand = new Random();
        rand.setSeed(SEED);
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

    //Support methods

    public static Vector strToVector(String str) {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++) {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data);
    }

    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException {
        if (Files.isDirectory(Paths.get(filename))) {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }
}
