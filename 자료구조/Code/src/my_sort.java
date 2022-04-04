import java.util.Arrays;

public class my_sort {
    //https://yabmoons.tistory.com/250

    private boolean desc;   //내림차순으로 할것인가
    private long runtime = 0L;
    private int QS_pivot = 2;   //0: 왼쪽 1:오른쪽 2:중간(default)
    private int[] sorted;   //merge sort에 필요한 임시 배열

    //O() = 최악, 평균, 최선

    public my_sort(){
        this.desc = false;
    }

    //Bubble_sort
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

    //Selection_sort
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

    //Quick_sort
    public void Quick_sort(int[] arr){
        //N^2, NlogN, NlogN
        //장점:기준값(Pivot)에 의한 분할을 통해 구현하는 정렬법 분할과정에서 logN 이라는 시간이 걸림 전체적으로 준수한 시간
        //단점:기준값(Pivot)에 따라서 시간복잡도는 날뛴다
        //기본적으로 재귀를 통해서 작동한다
        //테스트 결과: 전반적으로 속도는 중앙 < 왼쪽 ≒< 오른쪽 피벗 순으로 빠르다
        //10000000개 정렬 결과   19.466000(s)	24.667999(s)	0.416000(s)


        if(this.QS_pivot == -1) //값을 정해주지 않을 경우 랜덤으로 정한다
            this.QS_pivot =(int) (Math.random()*3); //0~2사이의 정수 = pivot 기준 왼쪽? 중간? 오른쪽? 랜덤 선택

        long beforeTime = System.currentTimeMillis();
        Quicksort(arr, 0, arr.length - 1);  //sorting 시작
        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
        
    }
    public void Quick_sort(int[] arr, boolean desc){
        this.desc = desc;
        this.Quick_sort(arr);
    }
    public void Quick_sort(int[] arr, boolean desc, int pivot) {
        try {
            if(pivot < 0 || pivot > 2)
                throw new Exception("pivot can't use " + pivot);
        }catch (Exception e){
            e.printStackTrace();
            System.exit(0);
        }
        this.QS_pivot = pivot;
        this.Quick_sort(arr, desc);
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
        if(left < right){   //원소가 1개만 남은 여부?
            int Pivot = Partition(arr, left, right);
            //(left, right 피펏은 마지막 swap을 통해 피벗이 들어갈 위치를 최종 선정 하기 때문에
            // 그 위치를 제외한 왼쪽(left ~ pivot_idx-1)과 오른쪽(pivot_idx+1 ~ right)으로 나누어 재귀를 해야하고
            // middle 피벗은 제자리 정렬(in-place sort)가 아니기 때문에 신중하게 생각해야한다
            // 따라서 리턴 받은 left ~ pivot_idx, pivot_idx+1 ~ right를 재귀호출 해야한다)
            if (this.QS_pivot == 2) {   //중간 피벗
                Quicksort(arr, left, Pivot);
            } else {    //left right 피벗
                Quicksort(arr, left, Pivot - 1);
            }
            Quicksort(arr, Pivot + 1, right);
        }
    }

    private int Partition(int[] arr, int left, int right){
        int l = left;
        int r = right;
        int pivot_v = 0;

        //Pivot_value설정(와 향상된 스위치문 뭐지? 멋지네)
        //왼쪽 피벗
        //왼쪽 피벗일때는 왼쪽(l)을 나중에 이동시켜 주어야 피벗이 정확한 위치(while문 종료 시점에 l이 무조건 작은 위치에 오게 된다)와 swap이 가능하다
        //l과 r은 각각 배열의 끝에서 1씩 벗어난 위치부터 시작한다
        switch (this.QS_pivot) {
            case 0 -> {
                pivot_v = arr[left];
                if (!this.desc) {
                    while (l < r) {
                        while (arr[r] > pivot_v && l < r) r--;

                        while (arr[l] <= pivot_v && l < r) l++;
                        swap(arr, l, r);
                    }
                } else {  //내림차순
                    while (l < r) {
                        while (arr[r] < pivot_v && l < r) r--;
                        while (arr[l] >= pivot_v && l < r) l++;
                        swap(arr, l, r);
                    }
                }
                swap(arr, left, r); //피벗값을 좌우로 나누어진 중간에 이동
                return r;
            }

            case 1 -> { //오른쪽 피벗
                pivot_v = arr[right];
                if (!this.desc) { //오름차순(default)
                    while (l < r) {
                        while (arr[l] < pivot_v && l < r) l++;
                        while (arr[r] >= pivot_v && l < r) r--;
                        swap(arr, l, r);
                    }
                } else {
                    while (l < r) {
                        while (arr[l] > pivot_v && l < r) l++;
                        while (arr[r] <= pivot_v && l < r) r--;
                        swap(arr, l, r);
                    }
                }
                swap(arr, right, l);    //피벗값을 좌우로 나누어진 중간에 이동
                return l;
            }

            case 2 -> { //중간 피벗
                l = left - 1;
                r = right + 1;
                pivot_v = arr[(left + right) / 2];
                if (!this.desc) { //오름차순(default)
                    while (true) {
                        while (arr[++l] < pivot_v) {}
                        while (arr[--r] > pivot_v && l <= r) {}   //가장 최신으로 갱신된 idx를 주의하라
                        if (l >= r) return r;
                        swap(arr, l, r);
                    }
                } else {    //내림차순
                    while (true) {
                        while (arr[++l] > pivot_v) {}
                        while (arr[--r] < pivot_v && l <= r) {}   //가장 최신으로 갱신된 idx를 주의하라
                        if (l >= r) return r;
                        swap(arr, l, r);
                    }
                }
            }
        }

        return 0;
    }


    //Heap Sort


    //Merge Sort
    public void Merge_sort(int[] arr){
        // O(NlogN), O(NlogN), O(NlogN)
        // 장점:항상 두 부분의 리스트로 쪼개서 재귀하기 때문에 최악의 경우에도 O(NlogN)유지, stable sort
        // 단점:정렬과정에서 추가적인 배열 공간 사용 메모리 사용량이 많다, 보조->원본 배열로 복사하는 과정에서 상대적으로 시간을 많이 소요
        long beforeTime = System.currentTimeMillis();

        this.sorted = new int[arr.length];
        merge_sort(arr, 0, arr.length - 1);
        this.sorted = null; //gc

        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
    }
    public void Merge_sort(int[] arr, boolean desc){
        this.desc = desc;
        this.Merge_sort(arr);
    }

    private void merge_sort(int[] arr, int left, int right){
        if(left == right)   return; //더이상 분할 할 수 없을때까지
        int mid = (left + right) / 2;   //중간 위치

        merge_sort(arr, left, mid);         //왼쪽 부분 리스트
        merge_sort(arr, mid+1, right);  //오른쪽 부분 리스트

        merge(arr, left, mid, right);   //병합 작업
    }
    /**
     * 
     * @param arr   정렬할 배열
     * @param left  배열의 시작점
     * @param mid   배열의 중간점
     * @param right 배열의 끝 점
     */
    private void merge(int[] arr, int left, int mid, int right){
        int l = left;       //왼쪽 시작점
        int r = mid + 1;    //오른쪽 시작점
        int idx = left; //채워넣을 배열의 인덱스

        if(!this.desc){
            while (l <= mid && r <= right) {    //한쪽이라도 다 끝날때까지 임시 배열에 정렬 순으로 넣기
                this.sorted[idx++] = (arr[l] <= arr[r]) ? arr[l++] : arr[r++];
            }
        }else{
            while (l <= mid && r <= right) {    //한쪽이라도 다 끝날때까지 임시 배열에 정렬 순으로 넣기
                this.sorted[idx++] = (arr[l] >= arr[r]) ? arr[l++] : arr[r++];
            }
        }


        //남은 부분들을 다 옮기기
        if(l > mid){
            while(r <= right){
                this.sorted[idx++] = arr[r++];
            }
        }else{
            while(l <= mid){
                this.sorted[idx++] = arr[l++];
            }
        }
        // 정렬된 배열은 원본 배열에 복사해서 옮겨준다

        /*
        src - 원본 배열
        srcPos - 원본 배열의 복사 시작 위치
        dest - 복사할 배열
        destPost - 복사할 배열의 복사 시작 위치
        length - 복사할 요소의 개수
         */
        //System.arraycopy(src, srcPos, dest, destPos, length);
        if (right + 1 - left >= 0) System.arraycopy(this.sorted, left, arr, left, right + 1 - left);
//        for(int i = left; i <= right; i++){
//            arr[i] = this.sorted[i];
//        }
    }

    //Insertion Sort
    public void Insertion_sort(int[] arr){
        //O(N^2), O(N^2), O(N)
        //장점:최선의 경우 O(N)의 빠른 효율성
        //단점:최학의 경우 O(N^2) 성능의 편차가 심하다
        long beforeTime = System.currentTimeMillis();

        for(int curr = 1; curr < arr.length; curr++){
            int target = arr[curr];
            int idx = curr - 1;

            if(!this.desc){
                while(idx >= 0 && arr[idx] > target){
                    arr[idx + 1] = arr[idx--];
                }
            }else{
                while(idx >= 0 && arr[idx] < target){
                    arr[idx + 1] = arr[idx--];
                }
            }

            arr[idx + 1] = target;
        }

        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
    }
    public void Insertion_sort(int[] arr, boolean desc){
        this.desc = desc;
        this.Insertion_sort(arr);
    }
    //Shell Sort
    //Radix Sort

    //Counting_sort
    public void Counting_sort(int[] arr){
        //O(N), O(N), O(N)
        //장점:압도적 속도 비교문도 필요 없다
        //단점:숫자가 큰 경우 그것에 상응하는 인덱스 번호가 필요하기 때문에 낭비되는 배열 메모리가 존재한다
        //stable 한가? => 개인적으로 생각한 결론은 counting한 결과로 배열을 재정립 하는 과정에 따라 달라진다
        //= orignal배열을 앞에서부터 loop문을 돌게되면 반대로(unstable) 뒤에서 돌게 되면 stable하게 정렬된다

        long beforeTime = System.currentTimeMillis();

        int[] result = new int[arr.length]; //reslut.length = 100(0~99)
        int[] mm = minmax(arr);     //최소 최대값 저장(최소값이 마이너스나 나올수도 있다)
        int[] cnt_arr = new int[mm[1] - mm[0] + 1]; //최소가 0 최대가 100일경우 배열의 크기는 101개가 필요하다
        for( int value : arr){
            cnt_arr[value - mm[0]]++;
        }
        for (int i = 1; i < cnt_arr.length; i++){
            cnt_arr[i] += cnt_arr[i-1];
        }
        for(int value : arr){
            int idx = value - mm[0];
            cnt_arr[idx]--;
            result[cnt_arr[idx]] = value;
        }
        arr = result;

        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
    }
    public void Counting_sort(int[] arr, boolean desc){
        this.desc = desc;
        this.Counting_sort(arr);
    }

    private int[] minmax(int[] arr){
        int[] res = new int[2];
        for(int value : arr){
            if (res[0] > value) res[0] = value;
            if (res[1] < value) res[1] = value;
        }
        return res;
    }

    
    private void swap(int[] arr, int i1, int i2){
        int temp = arr[i1];
        arr[i1] = arr[i2];
        arr[i2] = temp;
    }

    public long getRuntime(){
        return this.runtime;
    }

    public boolean isSort(int[] arr){
        if(!this.desc){ //오름차순 확인
            for(int i = 1; i < arr.length; i++){
                if(arr[i] < arr[i-1])   return false;
            }
        }   
        else{   //내림차순 확인
            for(int i = 1; i < arr.length; i++){
                if(arr[i] > arr[i-1])   return false;
            }
        }
        return true;
    }
}
