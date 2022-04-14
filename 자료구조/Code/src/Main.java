import java.util.Arrays;

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

    }


    public static void main(String[] args){
        list<Integer> arr = new list<>();
        for(int i = 0; i < 3; i++){
            arr.add(i);
        }
        System.out.println(arr.toString());


        arr.add(3, 3);
        System.out.println(arr.toString());


    }
}
