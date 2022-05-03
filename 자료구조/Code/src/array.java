import java.util.Arrays;    //배열을 가지고 놀수 있는 메서드들이 많이 들어있는 class? 라고 해야하나? ㅋ
import java.util.Scanner;


public class array {
    //속성(Attribute)(Field)
    //None

    //생성자(Constructor)
    array(){}

    //메서드(Method)

    //1차원 배열
    void arr_1D(){
        /*선언의 기본 문법==========================================================================
        DataType[] array_name;
        DataType array_name[];

        //선언된 배열은 new 키워드를 사용하여 실제 배열로 생성
        1.
        DataType[] array_name = new DataType[array_size];	//선호도가 높다
        2.
        DataType array_name[] = new DataType[array_size];
        3.
        DataType[] array_name                               //선언
        array_name = new DataType[array_size]               //new 연산자로 할당
         */

        //[1-1]:    int형 공간이 5개 있는 배열을 'i_arr'라는 이름으로 (선언+)할당 한다
        int[] i_arr1 = new int[5];
        i_arr1[0] = 3;   //배열의 시작 index는 0부터 출발한다
        i_arr1[1] = 11;
        i_arr1[2] = 22;
        i_arr1[3] = 33;
        i_arr1[4] = 44;  //index의 끝은 array_size - 1 이다

        System.out.println(i_arr1);  //그냥 print해버리면 배열의 주소값이 나옴(내가 원하는건 이게 아닌데)
        System.out.println(Arrays.toString(i_arr1)); //그래서 Arrays class가 준비 했다! toString() 메서드를 통해 인자 값들을 보기 좋게 출력해준다


        /*배열의 초기화=============================================================================
        DataType[] array_name = {element1, element2, ...};
        DataType[] array_name = new DataType[]{element1, element2, ...};
        -> 초기화 블록에 맞춰 자동으로 배열의 길이가 설정됨
         */
        //[1-2]: 선언과 동시에 초기화
        int[] i_arr2 = {55,66,77,88,99};                //아래의 방법(int[] i_arr3)보다는 편함(결과는 같음)
        int[] i_arr3 = new int[] {100,110,120,130,140};
        
        //[1-3]: 배열 선언 후 초기화
        int[] i_arr4;
        //i_arr4 = {1,2,3,4,5}; //Err: 배열의 시작 주소만 선언했을뿐 크기는 모르기 때문에 컴퓨터가 이해 못함
        i_arr4 = new int[] {1,2,3,4,5};


        //깨알 Tip================================================================================
        double[] d_arr1 = {9.8,2.45,5.98,5.11,3.14};    //당연하겠지만 다른 데이터 타입도 가능하다

        //각 배열의 마지막 원소 출력(원소의 개수를 알 필요없다 왜냐 배열의 길이를 반환하는 메서드를 사용으로 해결)
        System.out.println(i_arr1[i_arr1.length-1]);    //여기서 .length -1을 한 이유는 앞에서 설명 했듯이 array index의 끝은 array_size - 1이기 때문이다
        System.out.println(d_arr1[d_arr1.length-1]);

        //배열 이름 자체를 출력하게 되면 주소값을 출력한다
        System.out.println(d_arr1); //[D@주소값

        //배열을 한꺼번에 선언
        int[] a,b;      //정수형 배열 (a랑 b)
        int c[], d[];   //정수형 (배열 a, 배열 b)
        int e[], f;     //정수형 (배열 e, f) ->정수형 배열 하나와 정수형 변수 하나
                        //-> 같은 class(형태)로 여러 변수이름을 선언하고싶으면 위의 2가지 방법을 사용

        //[!]:베열 복사
        //arraycopy()메서드 사용 --> System.arraycopy(원본배열명, 원본시작인덱스, 복사배열명, 복사시작엔덱스, 길이만큼);
        int[] origin, copyarr;
        origin = new int[] {0,1,2,3,4};
        copyarr = new int[] {0,1,2,3,4,5,6,7,8,9};

        System.out.println("\nSystem.arraycopy()");
        System.out.println("origin:" + Arrays.toString(origin));
        System.out.println("befor copyarray:" + Arrays.toString(copyarr));
        System.arraycopy(origin, 2, copyarr, 4, 3);
        System.out.println("after copyarray:" + Arrays.toString(copyarr));
    }

    //다차원 배열
    void multi_dimensional_array(){
        /*
        //기본 선언 문법
        DataType[][] array_name;
        DataType array_name[][];
        DataType[] array_name[];
         */

        /*
        //배열의 선언과 동시에 초기화
        DataType array_name[row][colum] = {
            {element[0][0], element[0][1],...},
            {element[1][0], element[1][1],...},
            {element[2][0], element[2][1],...},
            ...
        };
        */

        String[][] arr = {
            {"한국","미국","일본"},
            {"태국","베트남","필리핀"}
        };

        //row x colum
        System.out.println("행열=" + arr.length + "x" + arr[0].length);

        System.out.println(arr[0]); //1차 배열에 대한 주소값 출력
        System.out.println(arr[1]); //1차 배열에 대한 주소값 출력

        //enhanced for문
        for(String[] row : arr){
            for(String element : row){
                System.out.println(element + " ");
            }
            System.out.println();
        }

        //[!]:charAt()메서드 -> 해당 인덱스에 있는 값을 반환 -> 단어를 char 배열에 한글자씩 저장 가능
        System.out.println("charAt()메서드 사용 '베트남'에서 charAt(1)=" + arr[1][1].charAt(1));

        /*
        //가변배열(Dynamic array)
        DataType[][] array_name = new DataType[row_size][];
        array_name[0] = new DataType[colum_size1];
        array_name[1] = new DataType[colum_size2];
        array_name[2] = new DataType[colum_size3];
        
        ->가변배열도 마찬가지로 선언과 동시에 할당 가능
         */
        
        int[][] dy_arr ={
                {10, 20},
                {10,20,30,40},
                {10}
        };

        for(int[] ar : dy_arr){
            System.out.println(Arrays.toString(ar));
        }

        array_2D_UserInput_training();
    }

    private void array_2D_UserInput_training(){
        System.out.println("2차원 배열 연습해보기=====================");
        Scanner sc = new Scanner(System.in);

        System.out.print("행 열을 띄어쓰기로 구분하여 입력하고 [Enter]:");
        int row = sc.nextInt();
        int col = sc.nextInt();

        char[][] gameMap = new char[row][col];

        String[] strArr = new String[row];
        for(int i = 0; i < row; i++){
            System.out.println((i+1) + "번째 행에 입력할 문자 " + col + "개(열)를 차례대로 입력하고 [Enter]:");
            strArr[i] = sc.next();
            for(int j = 0; j < col; j++){
                gameMap[i][j] = strArr[i].charAt(j);
            }
        }

        for(char[] carr : gameMap){
            for(char ch : carr){
                System.out.print(ch + " ");
            }
            System.out.println();
        }

    }

    public void test(){
        int row = 10, col = 10;
        int[][] arr = new int[row][col];

        for(int i = 1; i < row; i++){
            for(int j = 1; j < col; j++){
                arr[i][j] = i * j;
            }
        }


        for(int i = 1; i < 10; i++){
            System.out.print("%d");
            for(int j = 1; j < 10; j++){
                System.out.printf("[%2d]", arr[i][j]);
            }
            System.out.println();
        }

        
//        for(int[] r : arr){
//            for(int c : r){
//                System.out.printf("[%2d]", c);
//            }
//            System.out.println();
//        }
    }

    //배열의 여러 활용 방법에 대해서는
    //http://www.tcpschool.com/java/java_array_application  참고


   //이후 추가할 내용 아래 사이트 부터
    //http://www.tcpschool.com/java/java_array_memory
    
}
