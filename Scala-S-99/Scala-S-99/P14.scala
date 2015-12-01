import scala.collection.mutable.ListBuffer


object P14{
	def duplicate(ls: List[Any]): List[Any] = {
		var out = new ListBuffer[Any]()
		for (elem <- ls){
			out += elem
			out += elem
		}
		return out.toList
  	}

	def main(args: Array[String]) {
		val list = List('a', 'a', 'a', 'b', 'c')
    	println(duplicate(list))
	}
}
