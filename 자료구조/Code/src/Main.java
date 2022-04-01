import java.util.Arrays;
class Person {
    String name;
    int height;
    int weight;

    Person(String name, int height, int weight){
        this.name = name;
        this.height = height;
        this.weight = weight;
    }
    Person(){
        this.name = "홍길동";
        this.height = 5;
        this.weight = 0;
    }
    public void info(){
        System.out.printf("이름:%s\n키:%d\n몸무게:%d\n", name, height, weight);
    }
}

public class Main {

    public static int[] make_random_array(int arr_size, int max_num){
        int [] arr = new int[arr_size];
        for(int i = 0; i < arr.length; i++){
            arr[i] = (int) (Math.random() * max_num);
        }
        return arr;
    }


    public static void my_sort_test(){
        int[] arr = null;
        my_sort ms = new my_sort();
        for(int i = 0; i < 10; i++){
            arr = make_random_array(10, 50);
            System.out.println("TestCase " + (i+1));
            System.out.println(Arrays.toString(arr));
            ms.Quick_sort(arr);
//        System.out.printf("Q_sort:\t%f(s)\n",(float)ms.getRuntime()/ 1000);
            System.out.println(Arrays.toString(arr) + ", " + ms.isSort(arr));
        }
    }



    public static void stub(int a){
        System.out.println(a);


    }

    public static void main(String[] args) {
//        Person p1 = new Person("전세환", 1,2);
//        System.out.println(p1.name);
//        Person p2 = new Person();
//        System.out.println(p2.name);
        
        my_sort_test();
    }



}
