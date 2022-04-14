import java.util.Arrays;

public class list<T>{
    private static final int DEFAULT_CAPACITY = 10;
    private static final Object[] EMPTY_ARRAY = {};
    private T[] array;
    private int size;

    list(){
        array = (T[]) new Object[DEFAULT_CAPACITY];
        this.size = 0;
    }

    private void resize(){
        int array_capacity = this.array.length;

        if(array.equals((T[]) EMPTY_ARRAY)){
            array = (T[]) EMPTY_ARRAY;
            return;
        }

        if(array_capacity == this.size){
            this.array = Arrays.copyOf(this.array, array_capacity * 2);
            return;
        }
        if(size < array_capacity / 2){
            this.array = Arrays.copyOf(this.array, Math.max(array_capacity / 2, DEFAULT_CAPACITY));
            return;
        }
    }


    T get(int index){
        return this.array[index];
    }
    private void addLast(T value){
        if(array.length == this.size)   resize();
        this.array[size++] = value;

    }
    public boolean add(T value){
        addLast(value);
        return true;
    }

    public void add(int index, T value){
        if(index < 0 || index > size) throw new IndexOutOfBoundsException();
        if(index == size)   addLast(value);
        else{
            if(this.size == this.array.length)    resize();
            for(int i = size-1; i <= index; i--){
                this.array[i+1] = this.array[i];
            }
//            if(this.size - index >= 0)   System.arraycopy(this.array, index, this.array, index+1, this.size-index);
            this.array[index] = value;
            size++;
        }
    }

    public T remove(int index){
        if(index < 0 || index >= size)  throw new IndexOutOfBoundsException();
        T element = this.array[index];

        for(int i = index; i < size-1; i++){
            this.array[i] = this.array[i+1];
        }
        size--;
        this.array[size] = null;
//        if(size - index >= 0){
//            System.arraycopy(array, index +1, array, index, size - index - 1);
//            this.array[size-1] = null;
//        }
        resize();
        return element;
    }
    boolean clear(){
        this.array = (T[]) this.EMPTY_ARRAY;
        return true;
    }

    int size(){
        return this.size;
    }

    public String toString(){
        StringBuilder sb = new StringBuilder();
        for(int i = 0; i < this.size; ++i){
            sb.append(array[i]).append(" ");
        }
        sb.delete(sb.length()-1, sb.length());
        return sb.toString();
    }



}