import java.util.Arrays;
class Person {
    String name;
    int height;
    int weight;

<<<<<<< HEAD
    Person(String name, int height, int weight){
        this.name = name;
        this.height = height;
        this.weight = weight;
    }
    Person(){
        this.name = "홍길동";
        this.height = 5;
        this.weight = 0;
=======

public class Main {
    public static int[] make_iarr(int arr_size, int max_value){
        int[] arr = new int[arr_size];
        for(int i = 0; i < arr_size; i++){
            arr[i] = (int)(Math.random() * max_value);
        }
        return arr;
    }


    public static void quick_sort_test(){
        int[] arr;  //오리지날 배열
        int[] buf;  //동일한 측정을 위한 임시 배열
        float[] time_sum = new float[3];    //좌우중간 시간측정
        int Test_case = 1;
        my_sort ms = new my_sort();
        for(int i = 0; i < Test_case; i++){
            arr = make_iarr(10, 100);  //새로운 랜덤 배열 생성
//            System.out.println(Arrays.toString(arr)); //오리지날 배열
            for(int j = 0; j < 3; j++){
                buf = arr;  //동일한 랜덤 배열을 기반으로
                ms.Quick_sort(buf, false, j);   //좌우중간 한번씩 돌아가면서 정렬
//                System.out.print(j+":");  //어떤 피벗
//                System.out.println(Arrays.toString(buf) );    //정렬 후 배열
//                System.out.println(); 
                time_sum[j] += (float)ms.getRuntime()/1000; //각 정렬마다 시간 측정값 더하기
//                System.out.println(ms.isSort(arr));   //정렬 여부
            }
//            System.out.println("\n");

        }
        System.out.println("왼쪽\t\t\t오른쪽\t\t중간\t");
        for(float value : time_sum) {   //속도 평균 구하기
            System.out.printf("%f(s)\t", value / (float) Test_case);
        }
        System.out.println();

>>>>>>> ce5de000a9d2cba8c158cd7977d7370d83a7bb19
    }
    public void info(){
        System.out.printf("이름:%s\n키:%d\n몸무게:%d\n", name, height, weight);
    }
}

public class Main {

<<<<<<< HEAD
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

=======
    public static void merge_sort_test(){
        int [] arr;
        int test_case = 3;
        int arr_size = 10;
        my_sort ms = new my_sort();
        for(int i = 0; i < test_case; i++){
            StringBuilder sb = new StringBuilder();
            sb.append("TestCase").append(i+1).append("\n");
            arr = make_iarr(arr_size, 100);
            sb.append(Arrays.toString(arr)).append("\n");
            ms.Merge_sort(arr, false);
            sb.append(Arrays.toString(arr)).append("\n");
//            sb.append(ms.isSort(arr)).append(" ");
            sb.append(String.format("%f(s)",(float)ms.getRuntime()/1000)).append("\n");

            System.out.println(sb);

        }
    }

    public static void insertion_sort_test(){
        int[] arr;
        int test_case =3;
        int arr_size = 100000;
        my_sort ms = new my_sort();
        for(int i = 0; i < test_case; i++){
            StringBuilder sb = new StringBuilder();
            sb.append("TestCase").append(i+1).append("\n");
            arr = make_iarr(arr_size, 100);
//            sb.append(Arrays.toString(arr)).append("\n");
            ms.Insertion_sort(arr);
//            sb.append(Arrays.toString(arr)).append("\n");
//            sb.append(ms.isSort(arr)).append(" ");
            sb.append(String.format("%f(s)",(float)ms.getRuntime()/1000)).append("\n");
>>>>>>> ce5de000a9d2cba8c158cd7977d7370d83a7bb19

            System.out.println(sb);
        }
    }

    public static void stub(int a){
        System.out.println(a);



    public static void main(String[] args) {
<<<<<<< HEAD
//        Person p1 = new Person("전세환", 1,2);
//        System.out.println(p1.name);
//        Person p2 = new Person();
//        System.out.println(p2.name);
        
        my_sort_test();
=======
//        stub();
//        quick_sort_test();
//        merge_sort_test();
//        insertion_sort_test();

>>>>>>> ce5de000a9d2cba8c158cd7977d7370d83a7bb19
    }




}
