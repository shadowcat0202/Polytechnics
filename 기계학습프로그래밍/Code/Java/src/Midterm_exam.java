import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class Midterm_exam {
    Midterm_exam(){}

    void JavaPrimitiveSize(){
        System.out.println("byte의 크기: " + Byte.SIZE);
        System.out.println("char의 크기: " + Character.SIZE);
        System.out.println("int의 크기 " + Integer.SIZE);
        System.out.println("long의 크기" + Long.SIZE);
        System.out.println("short의 크기 " + Short.SIZE);
        System.out.println("float의 크기 " + Float.SIZE);
        System.out.println("double의 크기 " + Double.SIZE);
    }

    void BassSalmon(){
//        String path = System.getProperty("user.dir");
//        System.out.println(path);
        FileReader fr;
        BufferedReader br;
        try{
            fr = new FileReader("../../dataset/salmon_bass_data.csv");
            br = new BufferedReader(fr);

            String line = br.readLine();
            StringBuilder sb;
            while((line = br.readLine()) != null){
                String[] parse = line.split(",");
                sb = new StringBuilder();
                for(String value : parse){
                    sb.append(value).append(" ");
                }
                System.out.println(sb);
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }


    }
}
