object P01{
	def getLast(list: List[Int]): Int={
		return list.last
	}

	def main(args: Array[String]) {
		val list = List(1, 2, 3)
    	println(getLast(list))
	}
}
