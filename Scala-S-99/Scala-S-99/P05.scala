object P05{

	def reverse(list: List[Any]): List[Any] = {
		if (list.length == 0){
			return list
		}
		return List(list(list.length - 1)) ::: reverse(list.slice(0, list.length-1))
	}

	def main(args: Array[String]) {
		val list = List(1, 2, 3)
    	println(reverse(list))
	}
}
