import java.util.Arrays;

public class Main {
    public static int arr_max(int[] arr){
        int max = arr[0];
        for(int value : arr){
            if(max < value) max = value;
        }
        return max;
    }

    public static int arr_max_idx(int[] arr){
        int idx = 0;
        int max = arr[0];
        for(int i = 1; i < arr.length; i++){
            if(max < arr[i])    idx = i;
        }
        return idx;
    }

    public static void sort_test(){
        int [] score = new int[20];
        for(int i = 0; i < score.length; i++){
            score[i] = (int) (Math.random() * 100);
        }
        my_sort ms = new my_sort();
        ms.Bubble_sort(score);
        System.out.println(Arrays.toString(score));
        System.out.printf("%f(s)",(float)ms.getRuntime()/1000);
        System.out.println("종료~");
    }
    public static void stub(){
        for(int i = 0; i < 20; i++){
            System.out.print((int) (Math.random()*3) + ", ");
        }
    }
    public static void main(String[] args){
//        stub();
        sort_test();
    }

}
