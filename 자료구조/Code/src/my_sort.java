public class my_sort {
    //https://yabmoons.tistory.com/250
    private boolean desc;   //내림차순으로 할것인가
    private long runtime = 0L;
    private int QS_pivot = 0;   //0: 왼쪽 1:중간 2:오른쪽

    //O() = 최악, 평균, 최선
    public my_sort(){
        this.desc = false;
    }

    public void Bubble_sort(int[] arr){
        //O(N^2), O(N^2), O(N^2)
        //장점:구현이 쉽다, 직관적이다
        //단점:비효율적이다 시간복잡도는 모두 N^2 (노답이다)
        long beforeTime = System.currentTimeMillis();
        this.desc = false;
        for(int i = 0; i < arr.length; i++){
            //배열의 크기만큼 0~N-1까지 탐색하면서 인접한 칸과 비교하여 swap하는 방식
            for(int j = 0; j < arr.length - 1; j++){
                if(desc && (arr[j] > arr[j + 1]))    swap(arr, j, j + 1);
                if(!desc && (arr[j] < arr[j+1]))     swap(arr, j, j + 1);
            }
        }
        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
    }
    public void Bubble_sort(int[] arr, boolean desc){
        this.desc = desc;
        this.Bubble_sort(arr);
    }

    public void Selection_sort(int[] arr){
        //O(N^2), O(N^2), O(N^2)
        //장점:정렬을 위한 비교횟수는 많지만 실제 교환 횟수는 적기때문에 많은 교환을 요구하는 자료 상태에서는 그나마 효율적이다
        //단점:버블정렬과 같다(답도 없다)
        long beforeTime = System.currentTimeMillis();
        for(int i = 0; i < arr.length - 1; i++){
            int idx = i;    //시작은 기준인덱스부터 닷!
            for(int j = i + 1; j < arr.length; j++){
                //기준으로 잡은 인덱스를 시작으로 진행 방향의 모든 값들을 비교해서 최소|최대값이 존재하는 인덱스를 기억한다
                if(!this.desc && (arr[j] < arr[idx]))   idx = j;
                if(this.desc && (arr[j] > arr[idx]))     idx = j;
            }
            //기준 인덱스와 기억했던 인덱스의 차이가 발생(최소|최대의 위치가 그 자리가 아니다)하면 swap
            if(i != idx)    swap(arr, i, idx);
        }
        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
    }
    public void Selection_sort(int[] arr, boolean desc){
        this.desc = desc;   //기본속성 역할
        this.Selection_sort(arr);
        
    }

    public void Quick_sort(int[] arr){
        //N^2, NlogN, NlogN
        //장점:기준값(Pivot)에 의한 분할을 통해 구현하는 정렬법 분할과정에서 logN 이라는 시간이 걸림 전체적으로 준수한 시간
        //단점:기준값(Pivot)에 따라서 시간복잡도는 날뛴다
        //기본적으로 재귀를 통해서 작동한다
        this.QS_pivot =(int) (Math.random()*3); //0~2사이의 정수 = pivot 기준 왼쪽? 중간? 오른쪽? 랜덤 선택

        long beforeTime = System.currentTimeMillis();
        Quicksort(arr, 0, arr.length - 1);



        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
        
    }
    public void Quick_sort(int[] arr, boolean desc){
        this.desc = desc;
        this.Quick_sort(arr);
    }
    private void Quicksort(int[] arr, int left, int right){
        /*
        피벗을 기준으로 요소들을 왼쪽과 오른쪽으로 정렬
        Partitioning:
         arr[left]
        +-----------------------------------------------+
        |  pivot  |  elem <= pivot  |  element > pivot  |
        +-----------------------------------------------+

                           arr[lo]
        +-----------------------------------------------+
        |  elem <= pivot |  pivot  |   element > pivot  |
        +-----------------------------------------------+

        +----------------+         +--------------------+
        |  elem <= pivot |  pivot  |   element > pivot  |
        +----------------+         +--------------------+
        l          pivot-1         pivot+1              r
         */
        if(left < right){   //Pivot을 기준으로 왼쪽과 오른쪽이 나누어진것을 판단
            int Pivot = Partition(arr, left, right);
            Quicksort(arr, left, Pivot - 1);
            Quicksort(arr, Pivot + 1, right);
        }
    }
    private int Partition(int[] arr, int left, int right){
        int l = left;
        int r = right;
        int pivot_value = 0;

        switch (this.QS_pivot) {  //Pivot_value설정(와 향상된 스위치문 뭐지? 멋지네)
            case 0 :
                pivot_value = arr[left];
                break;
            case 1 :
                pivot_value = arr[(left + right) / 2];
                break;
            case 2 :
                pivot_value = arr[right];
                break;
        }

        while(l < r) {
            //왼쪽 피벗
            if (this.desc) {
                while (arr[r] <= pivot_value && l < r) r--;
                while (arr[l] > pivot_value && l < r) l++;
            } else {
                while (arr[r] > pivot_value && l < r) r--;
                while (arr[l] <= pivot_value && l < r) l++;
            }
            swap(arr, l, r);
        }

        switch(this.QS_pivot){
            case 0 :
                swap(arr, left, l);
                return l;   //swap을 진행헀다면 피벗요소는 l에 위치한다
            case 1 :
                swap(arr, left, l);
                return ;   //swap 진행헀다면 피벗요소는 l에 위치한다
            case 2 :
                swap(arr, right, r);
                return r;   //swap을 진행헀다면 피벗요소는 l에 위치한다
        //left와 right가 교차하기 전까지


            while(this.desc)
            //왼쪽피벗
            while(a[hi] > pivot && lo < hi) {
                hi--;
            }
            while(a[lo] <= pivot && lo < hi) {
                lo++;
            }

            //오른쪽피벗
            while(a[lo] < pivot && lo < hi) {
                lo++;
            }
            while(a[hi] >= pivot && lo < hi) {
                hi--;
            }

            //중간피벗
            do {
                lo++;
            } while(a[lo] < pivot);

            do {
                hi--;
            } while(a[hi] > pivot && lo <= hi);

            if(lo >= hi) {
                return hi;
            }

        }
        return 0;
    }
    
    private void swap(int[] arr, int i1, int i2){
        int temp = arr[i1];
        arr[i1] = arr[i2];
        arr[i2] = temp;
    }

    public long getRuntime(){
        return this.runtime;
    }
}
