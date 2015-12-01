object P02{

	def secondLast(list: List[Int]): Int = {
		return list(list.length - 2)
	}

	def main(args: Array[String]) {
		val list = List(1, 2, 3)
    	println(secondLast(list))
	}
}