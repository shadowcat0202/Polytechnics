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
        int [] arr = new int[500000000];
        for(int i = 0; i < arr.length; i++){
            arr[i] = (int) (Math.random() * 100);
        }
        int[] buf = arr;
//        System.out.println(Arrays.toString(arr));
        my_sort ms = new my_sort();
        arr = ms.Counting_sort(arr);
        System.out.printf("C_sort:\t%f(s)\n",(float)ms.getRuntime()/ 1000);
        System.out.println(ms.isSort(arr));
//
//        System.out.println(Arrays.toString(arr));
    }

    public static void Qick_sort(int[] arr, int left, int right){
        int pivot = parition_stub(arr, left, right);
        Qick_sort(arr, left, pivot - 1);
        Qick_sort(arr, pivot + 1, right);
    }

    public static int parition_stub(int[] arr, int left, int right){
        int l = left;
        int r = right;
        int pivot = (left + right) / 2;

        while(l <= r){
            while(arr[l] < arr[pivot]) l++;
            while(arr[r] > arr[pivot]) r--;
            if(l <= r){
                swap(arr, l, r);
                l++;
                r--;
            }
        }
        return l;
    }
    public static void swap(int[] arr, int i1, int i2){
        int temp = arr[i1];
        arr[i1] = arr[i2];
        arr[i2] = temp;
    }

    public static void stub(){
//        int[] arr = {1,2,3,4,5,6,7,8,9,10};
        int[] arr = new int[20];
        for(int i = 0; i < arr.length; i++){
            arr[i] = (int) (Math.random() * 100);
        }
//        System.out.println(Arrays.toString(arr));
//
//        System.out.println(Arrays.toString(arr));
//        System.out.println(res);



    }

    public static void main(String[] args) {
//        stub();
        sort_test();
    }



}
