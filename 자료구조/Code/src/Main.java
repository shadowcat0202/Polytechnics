import java.util.Arrays;
import java.util.Collections;


public class Main {

    int add(int a, int b){
        return a + b;
    }


    float avg(Integer[] arr){
        float sum = 0.0f;
        for(int i : arr){
            sum += i;
        }
        //return Arrays.stream(arr).sum() / arr.length;
        return sum / arr.length;
    }

    void test(Integer a, int[] b){
        System.out.println("in test()");

        System.out.println(++a);
        System.out.println(++b[0]);
    }
    static void swap(Integer a, Integer b){
        Integer tmp = a;
        a = b;
        b = tmp;
        System.out.println("내부");
        System.out.println(a + " " + b);
    }

    public static void main(String[] args) {
//        System.out.println(obj.add(3,5));
//        Integer[] arr = {1,5,6,72,3,74,1,5,6,7,2,6,6,4,7,8,1,6,5};
//        Arrays.sort(arr, Collections.reverseOrder());
//        System.out.println(arr[0] + "," + arr[arr.length-1]);
//        System.out.println(obj.avg(arr));
        Integer a = 1;
        Integer b = 2;
        System.out.println("전");
        System.out.println(a + " " + b);
        swap(a,b);
        System.out.println("후");
        System.out.println(a + " " + b);
    }


}
