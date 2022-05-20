import java.util.Arrays;
class Selection_sort{
    private void swap(int[] arr, int i1, int i2){
        if(i1 < 0 || i1 >= arr.length || i2 < 0 || i2 >= arr.length)  throw new ArrayIndexOutOfBoundsException();
        int temp = arr[i1];
        arr[i1] = arr[i2];
        arr[i2] = temp;
    }

    int argMin(int[] a, int start_idx){
        if(start_idx < 0 || start_idx >= a.length)  throw new ArrayIndexOutOfBoundsException();

        int min_value = a[start_idx];
        int min_idx = start_idx;

        for(int i = start_idx; i < a.length; i++){
            if(min_value > a[i]){
                min_value = a[i];
                min_idx = i;
            }
        }
        return min_idx;
    }
    int argMax(int[] a, int start_idx){
        if(start_idx < 0 || start_idx >= a.length)  throw new ArrayIndexOutOfBoundsException();

        int max_value = a[start_idx];
        int max_idx = start_idx;

        for(int i = start_idx; i < a.length; i++){
            if(max_value < a[i]){
                max_value = a[i];
                max_idx = i;
            }
        }
        return max_idx;
    }
    void selectionSort(int[] arr, boolean desc){
        int key = arr[0];
        int targetidx = 0;
        for(int i = 0; i < arr.length; i++){
            if(desc)  targetidx = argMin(arr, i);
            else targetidx = argMax(arr, i);
            swap(arr, i, targetidx);
            System.out.println(Arrays.toString(arr));
        }
    }
    void bubbleSort(int[] arr, boolean desc){
        for(int i = arr.length-1; i >= 0; i--){
            //배열의 크기만큼 0~N-1까지 탐색하면서 인접한 칸과 비교하여 swap하는 방식
            for(int j = 0; j < i; j++){
                if(!desc){
                    if(arr[j] > arr[j + 1]) swap(arr, j, j + 1);
                }else{
                    if(arr[j] < arr[j + 1]) swap(arr, j, j + 1);
                }
            }
        }
    }
}
public class sort_during_in_class {
    public static void main(String[] args){
        Selection_sort ss = new Selection_sort();
        int[] data = {69, 10, 30, 2, 16, 8, 31, 22};
//        ss.selectionSort(data, true);
        ss.bubbleSort(data, false);
        System.out.println(Arrays.toString(data));

    }

}


