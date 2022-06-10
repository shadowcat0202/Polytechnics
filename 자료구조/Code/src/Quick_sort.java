public class Quick_sort {
    private void swap(int[] arr, int idx1, int idx2){
        int tmp = arr[idx1];
        arr[idx1] = arr[idx2];
        arr[idx2] = tmp;
    }
    public void quickSort(int[] arr, int start, int end, boolean desc){
        if(start >= end)    return;
//        int pivot = start;
//        int pivot = (start + end) / 2;
        int pivot = end;

        int left = start;
        int right = end;
        if(!desc){  // 오름차순
            while(right >= left){   // 엇갈릴 때까지
                while(arr[left] < arr[pivot])   left++;
                while(arr[right] > arr[pivot])  right--;
                if(left < right){   // 교차하지 않았다면 변경
                    swap(arr, left, right);
                    left++;
                }
                else    break;
            }
        }else{  // 내림차순
            while(right >= left){   // 엇갈릴 때까지
                while(arr[left] > arr[pivot])   left++;
                while(arr[right] < arr[pivot])  right--;
                if(left < right){   // 교차하지 않았다면 변경
                    swap(arr, left, right);
                    left++;
                }
                else    break;
            }
        }

        swap(arr, pivot, right);    // 오른쪽 인덱스를 가장 최근에 갱신 했기 때문에 오른쪽이랑 피벗이랑 변경해야한다
        quickSort(arr, 0, right - 1, desc);
        quickSort(arr, right + 1, end, desc);
    }
}
