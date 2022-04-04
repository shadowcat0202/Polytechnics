import java.util.Arrays;
public class function {
<<<<<<< HEAD
    public function(){}

    public static int arr_max(int[] arr){
        int max = arr[0];
        for(int value : arr){
            if(max < value) max = value;
        }
        return max;
    }

=======
    public static int arr_max(int[] arr){
        int max = arr[0];
        for(int value : arr){
            if(max < value) max = value;
        }
        return max;
    }

>>>>>>> ce5de000a9d2cba8c158cd7977d7370d83a7bb19
    public static int arr_max_idx(int[] arr){
        int idx = 0;
        int max = arr[0];
        for(int i = 1; i < arr.length; i++){
            if(max < arr[i])    idx = i;
        }
        return idx;
    }
}
