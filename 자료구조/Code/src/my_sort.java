import java.util.Arrays;

public class my_sort {
    //https://yabmoons.tistory.com/250
    private boolean desc;   //내림차순으로 할것인가
    private long runtime = 0L;
    private int QS_pivot = 0;   //0: 왼쪽 1:중간 2:오른쪽

    //O() = 최악, 평균, 최선
    public my_sort(){
        this.desc = false;
    }
    
    //버블 소트
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
    
    //선택 소트
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

    //퀵 소트
    public void Quick_sort(int[] arr){
        //N^2, NlogN, NlogN
        //장점:기준값(Pivot)에 의한 분할을 통해 구현하는 정렬법 분할과정에서 logN 이라는 시간이 걸림 전체적으로 준수한 시간
        //단점:기준값(Pivot)에 따라서 시간복잡도는 날뛴다
        //기본적으로 재귀를 통해서 작동한다
//        this.QS_pivot =(int) (Math.random()*3); //0~2사이의 정수 = pivot 기준 왼쪽? 중간? 오른쪽? 랜덤 선택
        this.QS_pivot = 2;
        long beforeTime = System.currentTimeMillis();
        Quicksort(arr, 0, arr.length - 1);  //sorting 시작
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
        if(left < right){   //원소가 1개만 남은 여부?
            int Pivot = Partition(arr, left, right);
            if(this.QS_pivot == 2)  Quicksort(arr, left, Pivot);
            else    Quicksort(arr, left, Pivot-1);
            Quicksort(arr, Pivot + 1, right);
        }
    }
    private int Partition(int[] arr, int left, int right){
        int l = left;
        int r = right;
        int pivot_v = 0;

        //Pivot_value설정(와 향상된 스위치문 뭐지? 멋지네)
        //왼쪽 피벗
        switch (this.QS_pivot) {
            case 0 -> {
                //왼쪽 피벗일때는 왼쪽(l)을 나중에 이동시켜 주어야 피벗이 정확한 위치(while문 종료 시점에 l이 무조건 작은 위치에 오게 된다)와 swap이 가능하다
                pivot_v = arr[left];
                if (!this.desc) {
                    while (l < r) {
                        while (arr[r] > pivot_v && l < r) r--;
                        while (arr[l] <= pivot_v && l < r) l++; //후순위로 이동시켜야 조건문이 알맞게 떨어짐(left pivot일때는 r보다 l이 더 최신갱신이여야 한다)
                        swap(arr, l, r);
                    }
                }
                else{
                    while (l < r) {
                        while (arr[r] < pivot_v && l < r) r--;
                        while (arr[l] >= pivot_v && l < r) l++;
                        swap(arr, l, r);
                    }
                }
                swap(arr, left, l);
                return l;
            }
            case 1 -> { //오른쪽 피벗
                pivot_v = arr[right];
                //오른쪽 피벗일때는 오른쪽(r) 나중에 이동시켜 주어야 피벗이 정확한 위치(while문 종료 시점에 l이 무조건 작은 위치에 오게 된다)와 swap이 가능하다
                if(!this.desc){
                    while(l < r){
                        while (arr[l] < pivot_v && l < r) l++;
                        while (arr[r] >= pivot_v && l < r) r--; //후순위로 이동시켜야 조건문이 알맞게 떨어짐(Right pivot일때는 l보다 r이 더 최신갱신이여야 한다)
                        swap(arr, l, r);
                    }
                }
                else{
                    while(l < r){
                        while (arr[l] > pivot_v && l < r) l++;
                        while (arr[r] <= pivot_v && l < r) r--;
                        swap(arr, l, r);
                    }
                }
                swap(arr, right, r);
                return r;
            }
            case 2 -> { //중앙 피벗
                l = left - 1;
                r = right + 1;
                pivot_v = arr[(left + right) / 2];
                if(!this.desc){
                    while(true){
                        while(arr[++l] < pivot_v){}
                        while(arr[--r] > pivot_v && l <= r){}
                        if(l >= r)  return r;   //엇갈린다면 r를 반환
                        swap(arr, l, r);
                    }
                }
                else{
                    while(true){
                        while(arr[++l] > pivot_v){}
                        while(arr[--r] < pivot_v && l <= r){}
                        if(l >= r)  return r;   //엇갈린다면 l를 반환
                        swap(arr, l, r);
                    }
                }
            }
        }

        return 0;
    }

    //카운팅 소트
    public int[] Counting_sort(int[] arr){
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

        long afterTime = System.currentTimeMillis();
        this.runtime = (afterTime - beforeTime);
        return result;
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
