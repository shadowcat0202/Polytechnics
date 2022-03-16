import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;

public class bass_Salmon {
    private int Column_size;
    private int Row_size;
    private String Path;
    private String[] Column_name;

    public bass_Salmon(String path){
        this.Path = path;
    }

    public void read_data_draw_histogram(){
        int[] length_arr = null;   //물고기의 길이
        float[] brightness_arr = null; //물고기 밝기
        
        try{
            FileReader fr = new FileReader(Path);
            BufferedReader br = new BufferedReader(fr);
            String line = br.readLine();
            Column_name = line.split(",");  //컬럼 이름 저장(필요한건가?)
            Column_size = Column_name.length;


            Queue<String[]> tmp = new LinkedList<>();
            while((line = br.readLine()) != null){
                tmp.add(br.readLine().split(","));
            }
            Row_size = tmp.size();
            
            //동적 할당
            length_arr = new int[Row_size]; 
            brightness_arr = new float[Row_size];

            int idx = 0;
            while(!tmp.isEmpty()){
                String[] now = tmp.poll();
                length_arr[idx] = Integer.parseInt(now[1]);
                brightness_arr[idx++] = Float.parseFloat(now[2]);
            }
            tmp = null;      
        }catch (FileNotFoundException e) {
            e.printStackTrace();
            System.out.println("읽을 파일을 찾지 못했습니다");
        } catch (IOException e) {
            e.printStackTrace();
        }

        int[] length_minmax = get_minmax(length_arr);
        System.out.println(Arrays.toString(length_minmax));
        float[] brightness_minmax = get_minmax(brightness_arr);
        System.out.println(Arrays.toString(brightness_minmax));

        //히스토그램을 위한 배열 만들기
        int[] histo_length = get_histogram(length_arr, length_minmax);
        int[] histo_brigthness = get_histogram(brightness_arr, brightness_minmax);

        //히스토 그램 그려주기
        print_histogram_i(histo_length, length_minmax[0]);
        print_histogram_f(histo_brigthness, brightness_minmax[0]);


    }
    
    //최대 최소 잡아주는 메서드
    private <T> T get_minmax(T arr){
        if(arr instanceof int[]){
            int min = Integer.MAX_VALUE;
            int max = Integer.MIN_VALUE;
            for(int i = 0; i < ((int[]) arr).length; i++){
                if(((int[])arr)[i] < min)   min = ((int[])arr)[i];
                if(((int[])arr)[i] > max)   max = ((int[])arr)[i];
            }
            return (T) new int[] {min, max};
        }

        if(arr instanceof float[]){
            float min = Float.MAX_VALUE;
            float max = Float.MIN_VALUE;

            for(int i = 0; i < ((float[]) arr).length; i++){
                if(((float[])arr)[i] < min)   min = ((float[])arr)[i];
                if(((float[])arr)[i] > max)   max = ((float[])arr)[i];
            }
            return (T) new float[] {min, max};
        }
        return null;
    } 

    //히스토그램 배열을 구해주는 메서드
    public <T> int[] get_histogram(T arr, T mM){
        int[] histogram = null;
        if(arr instanceof int[]){
            histogram = new int[((int[])mM)[1] - ((int[])mM)[0]+1]; //길이가 2 -> [0], 26 -> [24] (index는 0번부터 시작하는다는 것을 잊지 말자!)
            for(int i = 0; i < ((int[]) arr).length; i++){
                histogram[((int[])arr)[i] - ((int[])mM)[0]]++;
            }
            return histogram;
        }
        if(arr instanceof float[]){
            histogram = new int[(int)((((float[])mM)[1] - ((float[])mM)[0]) * 10) + 1];
            for(int i = 0; i < ((float[]) arr).length; i++){
                histogram[(int) (((float[])arr)[i]*10 - ((float[])mM)[0]*10)]++;
            }
            return histogram;
        }
        return null;
        
    }

    //히스토그램을 그려주는 함수
    private void print_histogram_i(int[] histogram, int min){
        System.out.println("*****************************************************");
        for(int i = 0; i < histogram.length; i++){
            if(histogram[i] > 0){
                System.out.printf("%2d\t\t|", (i + min));
                for(int j = 0; j < histogram[i]; j++){
                    System.out.print("■");
                }
                System.out.println();
            }
        }
    }

    //히스토그램 그려주는 함수
    private void print_histogram_f(int[] histogram, float min){
        System.out.println("*****************************************************");
        for(int i = 0; i < histogram.length; i++){
            if(histogram[i] > 0){   //데이터가 없는건 무시
                System.out.printf("%.1f \t|", ((float)i / 10.0) + min);
                for(int j = 0; j < histogram[i]; j++){
                    System.out.print("■");
                }
                System.out.println();
            }
        }
    }
    
    public void test(){

    }
}
