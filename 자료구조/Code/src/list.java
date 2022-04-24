import java.util.Arrays;

public class list<T>{
    private T[] array;
    private static final int DEFAULT_SIZE = 3;
    private static final Object[] EMPTY_ARRAY = {};
    private int size;
    list(){
        array = (T[]) new Object[DEFAULT_SIZE];
        size = 0;
    }

    list(int make_size){
        array = (T[]) new Object[make_size];
        size = 0;
    }

    private void resize(){
        int array_capacity = this.array.length;
        if(this.array.equals(EMPTY_ARRAY)){
            array = (T[]) EMPTY_ARRAY;
            size = 0;
            return;
        }

        if(this.size == array_capacity){
            int new_capacity = array_capacity * 2;

            //방법1. 노가다
//            T[] temp = (T[]) new Object[new_capacity];
//            for(int i = 0; i < size; i++){
//                temp[i] =this.array[i];
//            }
//            this.array = temp;

            //방법2. 깔쌈하게 메서드 쓰기 Arrays.copyOf(array, new_array_capacity)
            Arrays.copyOf(array, new_capacity);
            return;
        }

        if(this.size < array_capacity / 2){
            int new_capacity = array_capacity / 2;

            //방법1. 노가다
//            T[] temp = (T[]) new Object[new_capacity];
//            for(int i = 0; i < size; i++){
//                temp[i] = this.array[i];
//            }
//            this.array = temp;

            //방법2. 깔쌈하게 메서드 쓰기 Arrays.copyOf(array, new_array_capacity)
            Arrays.copyOf((T[])array, new_capacity);
        }
    }

    private void addLast(T value){
        if(size == this.array.length)    resize();
        this.array[size] = value;
        size++;
    }

    public boolean add(T value){
        addLast(value);
        return true;
    }

    public boolean add(int index, T value){
        if(index < 0 || index >= this.array.length) throw new IndexOutOfBoundsException();
        if(size == index)
            addLast(value);
        else{
            if(this.size == array.length)   resize();

            //방법1. 노가다
//            for(int i = size; i > index; i--){
//                this.array[i] = this.array[i-1];
//            }

            //방법2. 함수 쓰자(물리적으로 시프트 시키기 때문에 빠르다)
            if (this.size - 1 - index >= 0) System.arraycopy(this.array, index, this.array, index + 1, this.size - 1 - index);

            this.array[index] = value;
            size++;
        }
        return true;
    }


    public T remove(int index){
        if(index < 0 || index >= this.size) throw new IndexOutOfBoundsException();

        T element =(T) this.array[index];   //지울 요소 반환 준비

        //방법1. 노가다
//        for(int i = index; i < this.size-1; i++){
//            this.array[i] = this.array[i+1];
//        }

        //방법2. 함수 쓰자(물리적으로 시프트 시키기 때문에 빠르다)
        if (this.size - 1 - index >= 0)   System.arraycopy(this.array, index + 1, this.array, index, this.size - 1 - index);

        this.array[size-1] = null;
        size--;
        return element;
    }
    
    public int size(){
        return this.size;
    }
    public void clear(){
        this.array = (T[]) EMPTY_ARRAY;
        this.size = 0;
    }

    public String toString(){
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        if(this.size > 0){
            for(int i = 0; i < this.size; ++i){
                sb.append(this.array[i]).append(" ");
            }
            sb.delete(sb.length()-1, sb.length());
        }
        sb.append("]");

        return sb.toString();
    }



}