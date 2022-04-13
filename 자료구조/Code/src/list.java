import java.util.Arrays;

public class list<T> {
    private int size;
    private T[] array;
    private static final int DEFAULT_CAPACITY = 10;
    private static final Object[] EMPTY_ARRAY = {};

    list(){
        this.array = (T[]) EMPTY_ARRAY;
        this.size = 0;
    }

    list(int capacity){
        this.array = (T[]) new Object[capacity];
        this.size = 0;
    }

    private void resize(){
        int array_capacity = array.length;

        if(Arrays.equals(array,EMPTY_ARRAY)){
            this.array =  (T[]) new Object[DEFAULT_CAPACITY];
            return;
        }

        if(size == array_capacity){
            int new_capacity = array_capacity * 2;
            this.array = Arrays.copyOf(this.array, new_capacity);
            return;
        }

        if(size < (array_capacity / 2)){
            int new_capacity = array_capacity / 2;
            this.array = Arrays.copyOf(this.array, Math.max(new_capacity, DEFAULT_CAPACITY));
            return;
        }
    }
    public boolean add(T value){
        addLast(value);
        return true;
    }

    private void addLast(T value) {
        if(size == array.length)    resize();
        array[size] = value;
        size++;
    }

    //특정 위치에 추가
    public void add(int index, T value){
        if(index > size || index < 0)   throw new IndexOutOfBoundsException();
        if(index == size)   addLast(value);
        else{
            if(size == array.length)    resize();

            if (size - index >= 0) System.arraycopy(array, index, array, index + 1, size - index);
//           for(int i = size; i > index; i--){
//               array[i] = array[i - 1];
//           }
            array[index] = value;
            size++;
        }
    }

    public T get(int index){
        return array[index];
    }
    public T remove(int index){
        if(index >= this.size || index < 0) throw new IndexOutOfBoundsException();

        T element = (T) this.array[index];
        this.array[index] = null;

        if(this.size - index >= 0){
            System.arraycopy(this.array, index+1, this.array, index, this.size - index -1);
            this.array[this.size-1] = null;
        }
        size--;
        resize();
        return element;
    }

    public boolean remove(T value){
        int index = indexOf(value);

        if(index == -1) return false;

        remove(index);
        return true;
    }

    private int indexOf(T value) {
        for(int i = 0; i < this.size; i++){
            if(this.array[i].equals(value)) return i;
        }
        return -1;
    }


    public void clear(){
        this.array = (T[]) EMPTY_ARRAY;
        this.size = 0;
    }

    public int size(){
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
